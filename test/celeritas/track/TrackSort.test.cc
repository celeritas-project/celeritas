
//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/TrackSort.test.cc
//---------------------------------------------------------------------------//
#include <algorithm>
#include <memory>
#include <vector>

#include "corecel/data/Collection.hh"
#include "corecel/io/LogContextException.hh"
#include "corecel/sys/ActionRegistry.hh"
#include "geocel/UnitUtils.hh"
#include "celeritas/TestEm3Base.hh"
#include "celeritas/Types.hh"
#include "celeritas/ext/GeantPhysicsOptions.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreTrackData.hh"
#include "celeritas/global/CoreTrackView.hh"
#include "celeritas/global/Stepper.hh"
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/Primary.hh"
#include "celeritas/track/TrackInitParams.hh"
#include "celeritas/track/detail/TrackSortUtils.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//
class TrackSortTestBase : virtual public GlobalTestBase
{
  public:
    using VecPrimary = std::vector<Primary>;

    //! Create a stepper
    template<MemSpace M>
    Stepper<M> make_stepper(size_type tracks)
    {
        CELER_EXPECT(tracks > 0);

        StepperInput result;
        result.params = this->core();
        result.stream_id = StreamId{0};
        result.num_track_slots = tracks;

        if constexpr (M == MemSpace::device)
        {
            device().create_streams(1);
        }

        return Stepper<M>{std::move(result)};
    }

    virtual VecPrimary make_primaries(size_type count) const = 0;

    template<MemSpace M>
    void step_action(std::string const& label, CoreState<M>& state)
    {
        ActionId action_id = this->action_reg()->find_action(label);
        CELER_VALIDATE(action_id, << "no '" << label << "' action found");
        auto action = dynamic_cast<CoreStepActionInterface const*>(
            this->action_reg()->action(action_id).get());
        CELER_VALIDATE(action, << "action '" << label << "' cannot execute");
        CELER_TRY_HANDLE(action->step(*this->core(), state),
                         LogContextException{this->output_reg().get()});
    }

    template<MemSpace M>
    void init_from_primaries(CoreState<M>& state, size_type num_primaries)
    {
        auto primaries = this->make_primaries(num_primaries);
        this->insert_primaries(state, make_span(primaries));
        this->step_action("extend-from-primaries", state);
        this->step_action("initialize-tracks", state);
    }

  private:
    // Primary initialization
    std::shared_ptr<ExtendFromPrimariesAction const> primaries_action_;
};

class TestEm3NoMsc : public TrackSortTestBase, public TestEm3Base
{
  public:
    GeantPhysicsOptions build_geant_options() const override
    {
        auto opts = TestEm3Base::build_geant_options();
        opts.msc = MscModelSelection::none;
        return opts;
    }

    //! Make a mix of 1 GeV electrons, positrons, and photons along +x
    VecPrimary make_primaries(size_type count) const override
    {
        Primary p;
        p.energy = units::MevEnergy{1000};
        p.position = from_cm({-22, 0, 0});
        p.direction = {1, 0, 0};
        p.time = 0;

        Array<ParticleId, 3> const particles = {
            this->particle()->find(pdg::electron()),
            this->particle()->find(pdg::positron()),
            this->particle()->find(pdg::gamma()),
        };
        CELER_ASSERT(particles[0] && particles[1] && particles[2]);

        std::vector<Primary> result(count, p);
        for (auto i : range(count))
        {
            result[i].event_id = EventId{i};
            result[i].particle_id = particles[i % particles.size()];
        }
        return result;
    }

  protected:
    auto build_init() -> SPConstTrackInit override
    {
        TrackInitParams::Input input;
        input.capacity = 4096;
        input.max_events = 4096;
        input.track_order = TrackOrder::reindex_step_limit_action;
        return std::make_shared<TrackInitParams>(input);
    }
};

#define TestTrackPartitionEm3Stepper \
    TEST_IF_CELERITAS_GEANT(TestTrackPartitionEm3Stepper)
class TestTrackPartitionEm3Stepper : public TestEm3NoMsc
{
  protected:
    auto build_init() -> SPConstTrackInit override
    {
        TrackInitParams::Input input;
        input.capacity = 4096;
        input.max_events = 4096;
        input.track_order = TrackOrder::reindex_status;
        return std::make_shared<TrackInitParams>(input);
    }
};

#define TestTrackSortActionIdEm3Stepper \
    TEST_IF_CELERITAS_GEANT(TestTrackSortActionIdEm3Stepper)
class TestTrackSortActionIdEm3Stepper : public TestEm3NoMsc
{
  protected:
    auto build_init() -> SPConstTrackInit override
    {
        TrackInitParams::Input input;
        input.capacity = 4096;
        input.max_events = 4096;
        input.track_order = TrackOrder::reindex_step_limit_action;
        return std::make_shared<TrackInitParams>(input);
    }
};

#define TestActionCountEm3Stepper \
    TEST_IF_CELERITAS_GEANT(TestActionCountEm3Stepper)
template<MemSpace M>
class TestActionCountEm3Stepper : public TestEm3NoMsc
{
  protected:
    using HostActionThreads =
        typename detail::CoreStateThreadOffsets<M>::HostActionThreads;
    using NativeActionThreads =
        typename detail::CoreStateThreadOffsets<M>::NativeActionThreads;
    using AllActionThreads = typename HostActionThreads::AllItemsT;

    auto build_init() -> SPConstTrackInit override
    {
        TrackInitParams::Input input;
        input.capacity = 4096;
        input.max_events = 4096;
        input.track_order = TrackOrder::reindex_step_limit_action;
        return std::make_shared<TrackInitParams>(input);
    }

    void check_action_count(HostActionThreads const& items, std::size_t size)
    {
        auto total_threads = 0;
        Span<ThreadId const> items_span = items[AllActionThreads{}];
        auto pos = std::find(items_span.begin(), items_span.end(), ThreadId{});
        ASSERT_EQ(pos, items_span.end());
        for (size_type i = 0; i < items.size() - 1; ++i)
        {
            Range<ThreadId> r{items[ActionId{i}], items[ActionId{i + 1}]};
            total_threads += r.size();
            ASSERT_LE(items[ActionId{i}], items[ActionId{i + 1}]);
        }
        ASSERT_EQ(total_threads, size);
    }
};

#define PartitionDataTest TEST_IF_CELERITAS_GEANT(PartitionDataTest)
class PartitionDataTest : public TestEm3NoMsc
{
  protected:
    auto build_init() -> SPConstTrackInit override
    {
        TrackInitParams::Input input;
        input.capacity = 4096;
        input.max_events = 128;
        input.track_order = TrackOrder::init_charge;
        return std::make_shared<TrackInitParams>(input);
    }

    template<MemSpace M>
    std::string get_result_string(CoreState<M> const& state)
    {
        auto const& params = this->core()->host_ref();
        auto sim_state = make_host_val(state.ref().sim);
        auto par_state = make_host_val(state.ref().particles);
        auto sim_ref = make_ref(sim_state);
        auto par_ref = make_ref(par_state);

        std::string result;
        for (size_type i = 0; i < state.size(); ++i)
        {
            TrackSlotId tsid{i};
            SimTrackView sim(params.sim, sim_ref, tsid);
            ParticleTrackView par(params.particles, par_ref, tsid);

            if (sim.status() == TrackStatus::inactive)
            {
                result += '_';
            }
            else if (par.particle_view().charge() == zero_quantity())
            {
                result += 'N';
            }
            else
            {
                result += 'C';
            }
        }
        return result;
    }
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(TestEm3NoMsc, host_is_sorting)
{
    CoreState<MemSpace::host> state{*this->core(), StreamId{0}, 128};

    this->init_from_primaries(state, state.size());
    this->step_action("pre-step", state);
    this->step_action("sort-tracks-post-step", state);
    auto track_slots = state.ref().track_slots.data();
    auto actions = detail::get_action_ptr(state.ref(),
                                          this->core()->init()->track_order());
    detail::ActionAccessor action_accessor{actions, track_slots};
    for (std::uint32_t i = 1; i < state.size(); ++i)
    {
        ASSERT_LE(action_accessor(ThreadId{i - 1}),
                  action_accessor(ThreadId{i}))
            << "Track slots are not sorted by action";
    }
}

TEST_F(TestTrackPartitionEm3Stepper, host_is_partitioned)
{
    // Create stepper and primaries, and take a step
    auto step = this->make_stepper<MemSpace::host>(128);
    auto primaries = this->make_primaries(8);
    step(make_span(primaries));

    auto check_is_partitioned = [&step] {
        auto span
            = step.state_ref().track_slots[AllItems<TrackSlotId::size_type>{}];
        return std::is_partitioned(
            span.begin(),
            span.end(),
            [&status = step.state_ref().sim.status](auto const track_slot) {
                return status[TrackSlotId{track_slot}] != TrackStatus::inactive;
            });
    };

    // we partition at the start of the step so we need to explicitly partition
    // again after a step before checking
    for (auto i = 0; i < 10; ++i)
    {
        detail::sort_tracks(step.state_ref(), TrackOrder::reindex_status);
        EXPECT_TRUE(check_is_partitioned()) << "Track slots are not "
                                               "partitioned by status";
        step();
    }
    step(make_span(primaries));
    for (auto i = 0; i < 10; ++i)
    {
        detail::sort_tracks(step.state_ref(), TrackOrder::reindex_status);
        EXPECT_TRUE(check_is_partitioned()) << "Track slots are not "
                                               "partitioned by status";
        step();
    }
}

TEST_F(TestTrackPartitionEm3Stepper,
       TEST_IF_CELER_DEVICE(device_is_partitioned))
{
    // Initialize some primaries and take a step
    auto step = this->make_stepper<MemSpace::device>(6400);
    auto primaries = this->make_primaries(8);
    step(make_span(primaries));

    auto check_is_partitioned = [&step] {
        // copy to host
        auto const& state_ref = step.state_ref();
        Collection<TrackSlotId::size_type, Ownership::value, MemSpace::host, ThreadId>
            track_slots;
        track_slots = state_ref.track_slots;
        StateCollection<TrackStatus, Ownership::value, MemSpace::host> track_status;
        track_status = state_ref.sim.status;

        // check for partitioned tracks
        auto span = track_slots[AllItems<TrackSlotId::size_type>{}];
        return std::is_partitioned(
            span.begin(), span.end(), [&track_status](auto const track_slot) {
                return track_status[TrackSlotId{track_slot}]
                       != TrackStatus::inactive;
            });
    };
    // we partition at the start of the step so we need to explicitly partition
    // again after a step before checking
    for (auto i = 0; i < 10; ++i)
    {
        detail::sort_tracks(step.state_ref(), TrackOrder::reindex_status);
        EXPECT_TRUE(check_is_partitioned()) << "Track slots are not "
                                               "partitioned by status";
        step();
    }
    step(make_span(primaries));
    for (auto i = 0; i < 10; ++i)
    {
        detail::sort_tracks(step.state_ref(), TrackOrder::reindex_status);
        EXPECT_TRUE(check_is_partitioned()) << "Track slots are not "
                                               "partitioned by status";
        step();
    }
}

TEST_F(TestTrackSortActionIdEm3Stepper, host_is_sorted)
{
    // Initialize some primaries and take a step
    auto step = this->make_stepper<MemSpace::host>(128);
    auto primaries = this->make_primaries(8);
    step(make_span(primaries));

    auto check_is_sorted = [&step] {
        auto& step_limit_action = step.state_ref().sim.post_step_action;
        auto& track_slots = step.state_ref().track_slots;
        for (celeritas::size_type i = 0; i < track_slots.size() - 1; ++i)
        {
            TrackSlotId tid_current{track_slots[ThreadId{i}]},
                tid_next{track_slots[ThreadId{i + 1}]};
            ActionId::size_type aid_current{
                step_limit_action[tid_current].unchecked_get()},
                aid_next{step_limit_action[tid_next].unchecked_get()};
            ASSERT_LE(aid_current, aid_next)
                << aid_current << " is larger than " << aid_next;
        }
    };
    // A step can change the step-limit action, so we need to redo the sorting
    // after taking a step.
    for (auto i = 0; i < 10; ++i)
    {
        detail::sort_tracks(step.state_ref(),
                            TrackOrder::reindex_step_limit_action);
        check_is_sorted();
        step();
    }
    step(make_span(primaries));
    for (auto i = 0; i < 10; ++i)
    {
        detail::sort_tracks(step.state_ref(),
                            TrackOrder::reindex_step_limit_action);
        check_is_sorted();
        step();
    }
}

TEST_F(TestTrackSortActionIdEm3Stepper, TEST_IF_CELER_DEVICE(device_is_sorted))
{
    // Initialize some primaries and take a step
    auto step = this->make_stepper<MemSpace::device>(6400);
    auto primaries = this->make_primaries(8);
    step(make_span(primaries));

    auto check_is_sorted = [&step] {
        // copy to host
        auto const& state_ref = step.state_ref();
        Collection<TrackSlotId::size_type, Ownership::value, MemSpace::host, ThreadId>
            track_slots;
        track_slots = state_ref.track_slots;
        StateCollection<ActionId, Ownership::value, MemSpace::host> step_limit;
        step_limit = state_ref.sim.post_step_action;

        for (celeritas::size_type i = 0; i < track_slots.size() - 1; ++i)
        {
            TrackSlotId tid_current{track_slots[ThreadId{i}]},
                tid_next{track_slots[ThreadId{i + 1}]};
            ActionId::size_type aid_current{
                step_limit[tid_current].unchecked_get()},
                aid_next{step_limit[tid_next].unchecked_get()};
            ASSERT_LE(aid_current, aid_next)
                << aid_current << " is larger than " << aid_next;
        }
    };

    // A step can change the step-limit action, so we need to redo the sorting
    // after taking a step.
    for (auto i = 0; i < 10; ++i)
    {
        detail::sort_tracks(step.state_ref(),
                            TrackOrder::reindex_step_limit_action);
        check_is_sorted();
        step();
    }
    step(make_span(primaries));
    for (auto i = 0; i < 10; ++i)
    {
        detail::sort_tracks(step.state_ref(),
                            TrackOrder::reindex_step_limit_action);
        check_is_sorted();
        step();
    }
}

using TestActionCountEm3StepperH = TestActionCountEm3Stepper<MemSpace::host>;
TEST_F(TestActionCountEm3StepperH, host_count_actions)
{
    // Initialize some primaries and take a step
    auto step = this->make_stepper<MemSpace::host>(128);
    auto primaries = this->make_primaries(8);
    step(make_span(primaries));

    // A step can change the step-limit action, so we need to redo the sorting
    // after taking a step.
    auto num_actions = this->action_reg()->num_actions();
    // can't access the collection in CoreState, so test do the counting in a
    // temporary instead
    HostActionThreads buffer;
    resize(&buffer, num_actions + 1);

    auto loop = [&] {
        detail::sort_tracks(step.state_ref(),
                            TrackOrder::reindex_step_limit_action);
        detail::count_tracks_per_action(step.state_ref(),
                                        buffer[AllActionThreads{}],
                                        buffer,
                                        TrackOrder::reindex_step_limit_action);

        check_action_count(buffer, step.state().size());
        step();
    };

    for (auto i = 0; i < 10; ++i)
    {
        loop();
    }

    step(make_span(primaries));

    for (auto i = 0; i < 10; ++i)
    {
        loop();
    }
}

using TestActionCountEm3StepperD = TestActionCountEm3Stepper<MemSpace::device>;
TEST_F(TestActionCountEm3StepperD, TEST_IF_CELER_DEVICE(device_count_actions))
{
    // Initialize some primaries and take a step
    auto step = this->make_stepper<MemSpace::device>(128);
    auto primaries = this->make_primaries(8);
    step(make_span(primaries));

    // A step can change the step-limit action, so we need to redo the sorting
    // after taking a step.
    auto num_actions = this->action_reg()->num_actions();
    // can't access the collection in CoreState, so test do the counting in a
    // temporary instead
    NativeActionThreads buffer_d;
    HostActionThreads buffer_h;
    resize(&buffer_d, num_actions + 1);
    resize(&buffer_h, num_actions + 1);

    auto loop = [&] {
        detail::sort_tracks(step.state_ref(),
                            TrackOrder::reindex_step_limit_action);
        detail::count_tracks_per_action(
            step.state_ref(),
            buffer_d[NativeActionThreads::AllItemsT{}],
            buffer_h,
            TrackOrder::reindex_step_limit_action);

        check_action_count(buffer_h, step.state().size());
        step();
    };

    for (auto i = 0; i < 10; ++i)
    {
        loop();
    }

    step(make_span(primaries));

    for (auto i = 0; i < 10; ++i)
    {
        loop();
    }
}

TEST_F(PartitionDataTest, init_primaries_host)
{
    // Initialize tracks from primaries and return a string representing the
    // location of the neutral and charged particles in the track vector
    {
        // 32 track slots and 2 primaries
        CoreState<MemSpace::host> state{*this->core(), StreamId{0}, 32};
        this->init_from_primaries(state, 2);
        auto result = this->get_result_string(state);
        EXPECT_EQ("______________________________CC", result);
    }
    {
        // 32 track slots and 16 primaries
        CoreState<MemSpace::host> state{*this->core(), StreamId{0}, 32};
        this->init_from_primaries(state, 16);
        auto result = this->get_result_string(state);
        EXPECT_EQ("NNNNN________________CCCCCCCCCCC", result);
    }
    {
        // 32 track slots and 32 primaries
        CoreState<MemSpace::host> state{*this->core(), StreamId{0}, 32};
        this->init_from_primaries(state, 32);
        auto result = this->get_result_string(state);
        EXPECT_EQ("NNNNNNNNNNCCCCCCCCCCCCCCCCCCCCCC", result);
    }
    {
        // 32 track slots and 64 primaries
        CoreState<MemSpace::host> state{*this->core(), StreamId{0}, 32};
        this->init_from_primaries(state, 64);
        auto result = this->get_result_string(state);
        EXPECT_EQ("NNNNNNNNNNNCCCCCCCCCCCCCCCCCCCCC", result);
    }
}

TEST_F(PartitionDataTest, TEST_IF_CELER_DEVICE(init_primaries_device))
{
    // Initialize tracks from primaries and return a string representing the
    // location of the neutral and charged particles in the track vector
    {
        // 8 track slots and 3 primaries
        CoreState<MemSpace::device> state{*this->core(), StreamId{0}, 8};
        this->init_from_primaries(state, 3);
        auto result = this->get_result_string(state);
        EXPECT_EQ("N_____CC", result);
    }
    {
        // 16 track slots and 17 primaries
        CoreState<MemSpace::device> state{*this->core(), StreamId{0}, 16};
        this->init_from_primaries(state, 17);
        auto result = this->get_result_string(state);
        EXPECT_EQ("NNNNNCCCCCCCCCCC", result);
    }
    {
        // 32 track slots and 30 primaries
        CoreState<MemSpace::device> state{*this->core(), StreamId{0}, 32};
        this->init_from_primaries(state, 31);
        auto result = this->get_result_string(state);
        EXPECT_EQ("NNNNNNNNNN_CCCCCCCCCCCCCCCCCCCCC", result);
    }
}

TEST_F(PartitionDataTest, step_host)
{
    // Take full steps in the transport loop and return a string representing
    // the location of the neutral and charged particles in the track vector at
    // the *end* of each step

    auto step = this->make_stepper<MemSpace::host>(64);
    auto primaries = this->make_primaries(4);

    std::vector<std::string> result;

    step(make_span(primaries));
    result.push_back(this->get_result_string(
        dynamic_cast<CoreState<MemSpace::host> const&>(step.state())));

    for (int i = 0; i < 20; ++i)
    {
        step();
        result.push_back(this->get_result_string(
            dynamic_cast<CoreState<MemSpace::host> const&>(step.state())));
    }
    if (this->is_ci_build())
    {
        static std::string const expected_result[] = {
            "N____________________________________________________________CCC",
            "_____________________________________________________________CCC",
            "NNN________________________________________________________CCCCC",
            "NNNNNNNN___________________________________________________CCCCC",
            "NNNNNNNNN__________________________________________________CCCCC",
            "N_NNNNNNNN________________________________________________CCCCCC",
            "_NNNNNNN_N_N_______________________________________________CCCCC",
            "N_NNNNNNNN_N________________________________________C__CCCCCCCCC",
            "NNNNNNNNNNNNN_N_____________________________________C__CCCCCCCCC",
            "NNNNN_N__N_NNNNNNNN________________________________CCCCCCCCCCCCC",
            "N_NNNNNNNNNNNNN_NNNNN______________________CCCC_CC_CCCCCCCCCCCCC",
            "NNNNNNNNNNNNNNNNNNNNNNNN_N_________________CCCCCCC_C_CCCCCCCCCCC",
            "NNNNNNNNNN_NNNNNN_NNNNNNNN_NNN_NN_NNN______CCCCCCCCCCC__CCCCCCCC",
            "NNNNNN_NNN_N__NNNNNNNNNNNNNNNNNN_NNNN_N____CCCCCCCCC_CCCCCCCCCCC",
            "NNNNNNNNNNNNN_NNNNNNNNNNNNNNNNNN_NNNN__NNNCCCCCCCC_CCCCCCCCCCCCC",
            "NNNNNNNNNNN_NNNNN_NNNNNNNNNNNNNNNNNNN_NNNN_CCCCCC_NC_CCCCCCCCCCC",
            "NNNNNNNNNNNNNNNNNN_NNNNNNNNN_NN_NNN_NNNNNNNCCCCCCNN_CC_CC_CCCCCC",
            "NNNNNNNNNNNNNNNN_N_NNNNNNNNN_NNNNNNNNNNNNNNCCC_CCNNN_CNCCCCCCCCC",
            "NN_N__NNNNN_NNNNNN_NNNNNNNNNNNN_NN_NNNNNNNNCCCNCC_NNCCN___CCCCCC",
            "NNNNN_NNNN_NNNNNN_NNNNNN_NNNNN_NNNNNNNNNNNNCCCNC_NNNCCNNCCCCCCCC",
            "NNNNN_NNNNNNNN_N_NNNNNNNNNNN_NNNNNNNNNNNNNNCCCNC_N_N_CNN__CCCCCC",
        };
        EXPECT_VEC_EQ(expected_result, result);
    }
}

TEST_F(PartitionDataTest, TEST_IF_CELER_DEVICE(step_device))
{
    // Take full steps in the transport loop and return a string representing
    // the location of the neutral and charged particles in the track vector at
    // the *end* of each step

    auto step = this->make_stepper<MemSpace::device>(64);
    auto primaries = this->make_primaries(32);

    std::vector<std::string> result;

    step(make_span(primaries));
    result.push_back(this->get_result_string(
        dynamic_cast<CoreState<MemSpace::device> const&>(step.state())));

    for (int i = 0; i < 20; ++i)
    {
        step();
        result.push_back(this->get_result_string(
            dynamic_cast<CoreState<MemSpace::device> const&>(step.state())));
    }
    if (this->is_ci_build())
    {
        static std::string const expected_result[] = {
            "NNNNNNNNNN________________________________CCCCCCCCCCCCCCCCCCCCCC",
            "_NNNNNNN_N________________________________CCCCCCCCCCCCCCCCCCCCCC",
            "NNNNNNNNNNNNN_NNNN_NNNNNNNNNN________CCCCCCCCCCCCCCCCCCCCCCCCCCC",
            "N_NNNNN_NNNNN_NNNNNNNNNNNNNNN_NNNN_NN_CCCCCCCCCCCCCCCCCCCCCCCCCC",
            "__NNNNNNNNNNNNNNN_NNNNNNNNNNNNNNNNNNNNCCCCCCCCCCCCCCCCCCCCCCCCCC",
            "NNNN_N_NNN_NNNNNNCNNNNNNNNNNNNNNNNNNNNCCCCCCCCCCCCCCCCCCCCCCCCCC",
            "N_NNNNN_NNNNNNNNN_NNNNNNNNNNNNNNNNNNNNCCCCCCCCCCCCCCCCCCCCCCCCCC",
            "NNNNN_NNNNNNNNNNN_NNNNNNNNNNNNNNNNNNNNCCCCCCCCCCCCCCCCCCCCCCCCCC",
            "_NNNNNNNNNN__NNNNNNN__NNNN__NNNNNNNNNNCCCCCCCCCCCCCCCCCCCCCCCCCC",
            "NNN_NNNNNN_NNNNNNNNNN_NNNNNNNNN_NNNNNNCCCCCCCCCCCCCCCCCCCCCCCCCC",
            "NNNNNNN_NNNNNNN_NNNNNNNNN_NNN_NNNNNNNNCCCCCCCCCCCCCCCCCCCCCCCCCC",
            "_NNNNNN_NNNNNNNNNNNNNNNNN_NNNCNNNN_NNNCCCCCCCCCCCCCCCCCCCCCCCCCC",
            "NNN_NNNNNN_NNNNNNNNNNNNNNCNNN_N_NNC_NNCCCCCCCCCCCCCCCCCCCCCCCCCC",
            "N_NNNNNNNNNNNNNNNNNNNNNNN_NNN__NNNCNNNCCCCCCCCCCCCCCCCCCCCCCCCCC",
            "NNNNNNNNNNNNN_NNNNNNNNNNNNNN_NNNNN_NNNCCCCCCCCCCCCCCCCCCCCCCCCCC",
            "NN_NNNNNNNNNNNNNNN_NNNNNN_NN_NNNNNNNNNCCCCCCCCCCCCCCCCCCCCCCCCCC",
            "NNNNNNN_NNNNNNNNNNNNNNNNNNNNNNNN_NNNN_CCCCCCCCCCCCCCCCCCCCCCCCCC",
            "NNNNNNNNNNNNNNN_NNNNNNNNNNNNNNNNNNNNNNCCCCCCCCCCCCCCCCCCCCCCCCCC",
            "N_NNNNNNNNNNNNNC_NNN_NNNNNNNNNNNNNNNNNCCCCCCCCCCCCCCCCCCCCCCCCCC",
            "NNNNNNNNN_NNNN_CNNNNNN_NNN_NNNNN_NNNNNCCCCCCCCCCCCCCCCCCCCCCCCCC",
            "NNNNNN_NN_NNNNNCNNNNNN_NNN_N_N_NCNNNNNCCCCCCCCCCCCCCCCCCCCCCCCCC",
        };
        EXPECT_VEC_EQ(expected_result, result);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
