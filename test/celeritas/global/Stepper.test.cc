//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/Stepper.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/global/Stepper.hh"

#include <memory>
#include <random>

#include "corecel/Config.hh"

#include "corecel/ScopedLogStorer.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"
#include "corecel/cont/Span.hh"
#include "corecel/data/AuxParamsRegistry.hh"
#include "corecel/io/LogContextException.hh"
#include "corecel/io/Logger.hh"
#include "corecel/random/engine/RngEngine.hh"
#include "corecel/sys/ActionRegistry.hh"
#include "geocel/UnitUtils.hh"
#include "celeritas/InvalidOrangeTestBase.hh"
#include "celeritas/SimpleTestBase.hh"
#include "celeritas/alongstep/AlongStepUniformMscAction.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/Primary.hh"
#include "celeritas/track/SimParams.hh"
#include "celeritas/track/SimTrackView.hh"
#include "celeritas/track/TrackInitParams.hh"

#include "DummyAction.hh"
#include "StepperTestBase.hh"
#include "celeritas_test.hh"

using celeritas::units::MevEnergy;

namespace celeritas
{
namespace test
{
namespace
{
//---------------------------------------------------------------------------//
void expect_stepper_eq(StepperResult const& expected,
                       StepperResult const& actual)
{
    EXPECT_EQ(expected.generated, actual.generated);
    EXPECT_EQ(expected.queued, actual.queued);
    EXPECT_EQ(expected.active, actual.active);
    EXPECT_EQ(expected.alive, actual.alive);
    EXPECT_EQ(expected.cut, actual.cut);
    EXPECT_EQ(expected.errored, actual.errored);
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class SimpleComptonTest : public SimpleTestBase, public StepperTestBase
{
  public:
    std::vector<Primary> make_primaries(size_type count) const override
    {
        Primary p;
        p.particle_id = this->particle()->find(pdg::gamma());
        CELER_ASSERT(p.particle_id);
        p.energy = units::MevEnergy{100};
        p.position = from_cm(Real3{-22, 0, 0});
        p.direction = {1, 0, 0};
        p.time = 0;
        p.event_id = EventId{0};

        std::vector<Primary> result(count, p);
        return result;
    }

    SPConstSim build_sim() override
    {
        SimParams::Input input;
        input.particles = this->particle();
        if (max_steps_ > 0)
        {
            input.max_steps = max_steps_;
        }
        CELER_LOG(debug) << "Building with step limit of " << input.max_steps;
        return std::make_shared<SimParams>(input);
    }

    size_type max_average_steps() const override { return 100000; }

    template<MemSpace M>
    void run_staged_first_step()
    {
        size_type num_primaries = 32;
        size_type num_tracks = 64;

        Stepper<M> step(this->make_stepper_input(num_tracks));
        auto primaries = this->make_primaries(num_primaries);
        step.stage_primaries(make_span(primaries));

        auto counters = step.state().sync_get_counters();
        EXPECT_EQ(num_primaries, counters.num_pending);

        auto result = step();
        EXPECT_EQ(num_primaries, result.active);
        EXPECT_GE(num_primaries, result.alive);

        counters = step.state().sync_get_counters();
        EXPECT_EQ(0, counters.num_pending);
    }

    template<MemSpace M>
    void run_buffered_pipeline()
    {
        size_type const num_primaries = 32;
        size_type const num_tracks = 64;

        Stepper<M> expected_step(this->make_stepper_input(num_tracks));
        auto first = this->make_primaries(num_primaries);
        auto second = this->make_primaries(num_primaries);
        auto const expected_first = expected_step(make_span(first));
        auto const expected_second = expected_step(make_span(second));

        Stepper<M> step(this->make_stepper_input(num_tracks));
        EXPECT_GE(step.primary_capacity(), num_primaries);
        EXPECT_EQ(0, step.num_buffered_primaries());
        EXPECT_TRUE(step.staged_primaries().empty());

        for (auto primary : first)
        {
            step.push_primary(primary);
        }
        EXPECT_EQ(num_primaries, step.num_buffered_primaries());
        step.stage_primaries();
        EXPECT_EQ(0, step.num_buffered_primaries());
        ASSERT_EQ(num_primaries, step.staged_primaries().size());
        auto const* first_data = step.staged_primaries().data();

        step.async();
        EXPECT_TRUE(step.valid());
        EXPECT_TRUE(step.staged_primaries().empty());

        for (auto primary : second)
        {
            step.push_primary(primary);
        }
        step.stage_primaries();
        ASSERT_EQ(num_primaries, step.staged_primaries().size());
        EXPECT_NE(first_data, step.staged_primaries().data());

        step.wait();
        EXPECT_TRUE(step.ready());
        expect_stepper_eq(expected_first, step.get());
        EXPECT_FALSE(step.valid());
        EXPECT_EQ(num_primaries, step.staged_primaries().size());

        step.async();
        expect_stepper_eq(expected_second, step.get());
        EXPECT_TRUE(step.staged_primaries().empty());
    }

    template<MemSpace M>
    void run_buffered_continuation()
    {
        size_type const num_primaries = 32;
        size_type const num_tracks = 8;

        Stepper<M> step(this->make_stepper_input(num_tracks));
        EXPECT_GE(step.initializer_capacity(), num_primaries);
        auto first = this->make_primaries(num_primaries);
        auto second = this->make_primaries(num_primaries);

        step.async(make_span(first));
        for (auto primary : second)
        {
            step.push_primary(primary);
        }

        step.wait();
        EXPECT_TRUE(step.ready());
        auto result = step.get();
        ASSERT_TRUE(result);
        for (size_type step_iters = 0; result; ++step_iters)
        {
            ASSERT_LT(step_iters, 10000);
            EXPECT_EQ(num_primaries, step.num_buffered_primaries());
            EXPECT_TRUE(step.staged_primaries().empty());

            step.async();
            result = step.get();
        }

        EXPECT_EQ(num_primaries, step.num_buffered_primaries());
        step.stage_primaries();
        step.async();
        result = step.get();
        EXPECT_EQ(num_primaries, result.generated);
    }

    size_type max_steps_{0};
};

class StepperOrderTest : public SimpleComptonTest
{
  public:
    void SetUp()
    {
        auto& aux_reg = this->aux_reg();
        auto& action_reg = this->action_reg();

        dummy_params_ = std::make_shared<DummyParams>(aux_reg->next_id());
        aux_reg->insert(dummy_params_);

        // Note that order shouldn't matter; we deliberately add these out of
        // order.
        for (auto action_order : {StepActionOrder::user_post,
                                  StepActionOrder::user_start,
                                  StepActionOrder::user_pre})
        {
            // Create a new action that can read data from the dummy params
            action_reg->insert(std::make_shared<DummyAction>(
                action_reg->next_id(),
                action_order,
                std::string("dummy-") + to_cstring(action_order),
                dummy_params_->aux_id()));
        }
    }

    DummyState const& get_dummy_state(
        CoreStateInterface const& core_state) const
    {
        return get<DummyState>(core_state.aux(), dummy_params_->aux_id());
    }

    std::shared_ptr<DummyParams> dummy_params_;
};

class AsyncStepperTest : public SimpleComptonTest
{
  public:
    void SetUp() { this->disable_status_checker(); }
};

#define BadGeometryTest TEST_IF_CELERITAS_ORANGE(BadGeometryTest)
class BadGeometryTest : public InvalidOrangeTestBase
{
  public:
    Primary make_primary(Real3 const& pos)
    {
        Primary p;
        p.particle_id = this->particle()->find(pdg::gamma());
        CELER_ASSERT(p.particle_id);
        p.energy = units::MevEnergy{100};
        p.event_id = EventId{0};
        p.position = pos;
        p.direction = {1, 0, 0};
        p.time = 0;
        return p;
    }

    StepperInput make_stepper_input()
    {
        StepperInput result;
        result.params = this->core();
        result.stream_id = StreamId{0};
        result.num_track_slots = 1;
        result.actions = std::make_shared<ActionSequence>(
            *this->action_reg(), ActionSequence::Options{});
        return result;
    }

    template<MemSpace M>
    ScopedLogStorer run_one_failure(Real3 const& point)
    {
        Stepper<M> step(this->make_stepper_input());

        auto primary = this->make_primary(point);
        ScopedLogStorer scoped_log{&celeritas::self_logger()};
        CELER_TRY_HANDLE(step({&primary, 1}),
                         LogContextException{this->output_reg().get()});

        {
            // Check that the state failed
            auto sim_state = make_host_val(step.state_ref().sim);
            auto state_ref = make_ref(sim_state);
            SimTrackView sim(
                this->core()->host_ref().sim, state_ref, TrackSlotId{0});
            EXPECT_EQ(this->core()->host_ref().scalars.tracking_cut_action,
                      sim.post_step_action());
        }
        return scoped_log;
    }
};

//---------------------------------------------------------------------------//
// Two boxes: compton with fake cross sections
//---------------------------------------------------------------------------//

TEST_F(SimpleComptonTest, setup)
{
    auto result = this->check_setup();
    static char const* expected_process[] = {"Compton scattering"};
    EXPECT_VEC_EQ(expected_process, result.processes);
}

TEST_F(SimpleComptonTest, fail_initialize)
{
    Stepper<MemSpace::host> step(this->make_stepper_input(32));

    auto primaries = this->make_primaries(16);
    primaries.back().position = from_cm({1001, 0, 0});
    {
        ScopedLogStorer scoped_log{&celeritas::self_logger()};
        CELER_TRY_HANDLE(step(make_span(primaries)),
                         LogContextException{this->output_reg().get()});

        if (CELERITAS_UNITS == CELERITAS_UNITS_CGS)
        {
            std::vector<std::string> expected_log_messages = {
                "Track 31 started outside the geometry",
                R"(Killing track {"geo":{"dir":[1.0,0.0,0.0],"is_on_boundary":false,"is_outside":true,"pos":[[1001.,0.0,0.0],"cm"]},"mat":null,"particle":{"energy":[100.0,"MeV"],"particle_id":"gamma"},"sim":{"event_id":0,"num_steps":0,"parent_id":null,"post_step_action":"tracking-cut","primary_id":null,"status":"errored","step_length":[0.0,"cm"],"time":[0.0,"s"],"track_id":15},"thread_id":31,"track_slot_id":31}: depositing 100 MeV)",
            };
            if (CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_VECGEOM)
            {
                expected_log_messages[0]
                    = "Failed to initialize geometry state at {1001, 0, 0} cm";
            }
            EXPECT_VEC_EQ(expected_log_messages, scoped_log.messages());
        }
        static char const* const expected_log_levels[] = {"error", "error"};
        EXPECT_VEC_EQ(expected_log_levels, scoped_log.levels());
    }

    // Check that the out-of-bounds track was killed
    auto const& core_scalars = this->core()->host_ref().scalars;
    auto const& sim_state = step.state_ref().sim;
    EXPECT_EQ(core_scalars.tracking_cut_action,
              sim_state.post_step_action[TrackSlotId{31}]);
}

TEST_F(SimpleComptonTest, host)
{
    size_type num_primaries = 32;
    size_type num_tracks = 64;

    Stepper<MemSpace::host> step(this->make_stepper_input(num_tracks));
    auto result = this->run(step, num_primaries);

    if (this->is_default_build())
    {
        EXPECT_EQ(919, result.num_step_iters());
        EXPECT_SOFT_EQ(53.8125, result.calc_avg_steps_per_primary());
        EXPECT_EQ(RunResult::StepCount({1, 6}), result.calc_queue_hwm());
    }
    EXPECT_EQ(3, result.calc_emptying_step());
}

TEST_F(SimpleComptonTest, TEST_IF_CELER_DEVICE(device))
{
    size_type num_primaries = 32;
    size_type num_tracks = 64;

    Stepper<MemSpace::device> step(this->make_stepper_input(num_tracks));
    auto result = this->run(step, num_primaries);
    if (this->is_default_build())
    {
        EXPECT_EQ(919, result.num_step_iters());
        EXPECT_SOFT_EQ(53.8125, result.calc_avg_steps_per_primary());
        EXPECT_EQ(RunResult::StepCount({1, 6}), result.calc_queue_hwm());
    }
    EXPECT_EQ(3, result.calc_emptying_step());
}

TEST_F(SimpleComptonTest, stage_primaries_host)
{
    this->run_staged_first_step<MemSpace::host>();
}

TEST_F(SimpleComptonTest, fail_stage_primaries_twice)
{
    size_type num_primaries = 32;
    size_type num_tracks = 64;

    Stepper<MemSpace::host> step(this->make_stepper_input(num_tracks));
    auto primaries = this->make_primaries(num_primaries);
    step.stage_primaries(make_span(primaries));

    EXPECT_THROW(step.stage_primaries(make_span(primaries)), RuntimeError);
}

TEST_F(SimpleComptonTest, recover_from_invalid_external_primaries)
{
    Stepper<MemSpace::host> step(this->make_stepper_input(64));
    auto primaries = this->make_primaries(1);
    primaries.front().event_id = EventId{this->init()->max_events()};

    EXPECT_THROW(step.stage_primaries(make_span(primaries)), RuntimeError);
    EXPECT_EQ(0, step.num_buffered_primaries());
    EXPECT_TRUE(step.staged_primaries().empty());

    primaries.front().event_id = EventId{0};
    EXPECT_NO_THROW(step.stage_primaries(make_span(primaries)));
    EXPECT_EQ(1, step.staged_primaries().size());
    static_cast<void>(step());
}

TEST_F(SimpleComptonTest, recover_from_initializer_capacity_failure)
{
    size_type const init_capacity = this->init()->capacity();
    auto input = this->make_stepper_input(64);
    input.primary_capacity = init_capacity + 1;
    Stepper<MemSpace::host> step(std::move(input));
    auto primaries = this->make_primaries(init_capacity + 1);

    EXPECT_THROW(step.stage_primaries(make_span(primaries)), RuntimeError);
    EXPECT_EQ(0, step.num_buffered_primaries());
    EXPECT_TRUE(step.staged_primaries().empty());
    EXPECT_EQ(0, step.state().sync_get_counters().num_pending);

    EXPECT_NO_THROW(step.stage_primaries(make_span(primaries).first(1)));
    EXPECT_EQ(1, step.staged_primaries().size());
    EXPECT_EQ(1, step.state().sync_get_counters().num_pending);
    static_cast<void>(step());
}

TEST_F(SimpleComptonTest, primary_capacity_override)
{
    auto input = this->make_stepper_input(64);
    input.primary_capacity = 2;
    Stepper<MemSpace::host> step(std::move(input));
    EXPECT_EQ(2, step.primary_capacity());
    EXPECT_EQ(step.state_ref().physics.secondaries.capacity(),
              step.secondary_capacity());

    auto primaries = this->make_primaries(3);
    EXPECT_THROW(step.stage_primaries(make_span(primaries)), RuntimeError);
    step.stage_primaries(make_span(primaries).first(2));
    EXPECT_EQ(2, step.staged_primaries().size());
    static_cast<void>(step());
}

TEST_F(SimpleComptonTest, TEST_IF_CELER_DEVICE(stage_primaries_device))
{
    this->run_staged_first_step<MemSpace::device>();
}

TEST_F(SimpleComptonTest, buffered_pipeline_host)
{
    this->run_buffered_pipeline<MemSpace::host>();
}

TEST_F(SimpleComptonTest, TEST_IF_CELER_DEVICE(buffered_pipeline_device))
{
    this->run_buffered_pipeline<MemSpace::device>();
}

TEST_F(SimpleComptonTest, buffered_continuation_host)
{
    this->run_buffered_continuation<MemSpace::host>();
}

TEST_F(SimpleComptonTest, TEST_IF_CELER_DEVICE(buffered_continuation_device))
{
    this->run_buffered_continuation<MemSpace::device>();
}

TEST_F(SimpleComptonTest, fail_queued_primary_operations)
{
    Stepper<MemSpace::host> step(this->make_stepper_input(64));
    auto primaries = this->make_primaries(1);

    EXPECT_THROW(step.stage_primaries(), RuntimeError);
    step.push_primary(primaries.front());
    EXPECT_THROW(step.warm_up(), RuntimeError);
    EXPECT_THROW(step.reset_state(), RuntimeError);
    EXPECT_THROW(step.reseed(UniqueEventId{123}), RuntimeError);
    EXPECT_NO_THROW(step.kill_active());

    step.stage_primaries();
    EXPECT_THROW(step.stage_primaries(), RuntimeError);
    EXPECT_THROW(step.stage_primaries(make_span(primaries)), RuntimeError);
    EXPECT_THROW(step.warm_up(), RuntimeError);
    EXPECT_THROW(step.reset_state(), RuntimeError);
    EXPECT_THROW(step.reseed(UniqueEventId{123}), RuntimeError);
    EXPECT_THROW(step.kill_active(), RuntimeError);

    step.async();
    static_cast<void>(step.get());
}

TEST_F(SimpleComptonTest, async_lifecycle_host)
{
    size_type const num_primaries = 32;
    size_type const num_tracks = 64;

    Stepper<MemSpace::host> expected_step(this->make_stepper_input(num_tracks));
    Stepper<MemSpace::host> step(this->make_stepper_input(num_tracks));
    auto primaries = this->make_primaries(num_primaries);

    EXPECT_FALSE(step.valid());
    EXPECT_THROW(step.ready(), RuntimeError);
    EXPECT_THROW(step.wait(), RuntimeError);
    EXPECT_THROW(step.get(), RuntimeError);

    auto expected_result = expected_step(make_span(primaries));
    step.async(make_span(primaries));
    EXPECT_TRUE(step.valid());
    EXPECT_TRUE(step.ready());
    step.wait();
    EXPECT_TRUE(step.valid());
    auto result = step.get();
    expect_stepper_eq(expected_result, result);
    EXPECT_FALSE(step.valid());

    expected_result = expected_step();
    step.async();
    EXPECT_TRUE(step.valid());
    EXPECT_TRUE(step.ready());
    EXPECT_NO_THROW(step.wait());
    EXPECT_TRUE(step.valid());
    EXPECT_THROW(step.async(), RuntimeError);
    EXPECT_THROW(step.async(make_span(primaries)), RuntimeError);
    EXPECT_THROW(step.warm_up(), RuntimeError);
    EXPECT_THROW(step.reset_state(), RuntimeError);
    EXPECT_THROW(step.reseed(UniqueEventId{123}), RuntimeError);
    EXPECT_THROW(step.kill_active(), RuntimeError);
    EXPECT_THROW(step(make_span(primaries)), RuntimeError);

    result = step.get();
    expect_stepper_eq(expected_result, result);
    EXPECT_FALSE(step.valid());
    EXPECT_THROW(step.ready(), RuntimeError);
    EXPECT_THROW(step.wait(), RuntimeError);
    EXPECT_THROW(step.get(), RuntimeError);
}

TEST_F(AsyncStepperTest, TEST_IF_CELER_DEVICE(async_lifecycle_device))
{
    size_type const num_primaries = 32;
    size_type const num_tracks = 64;

    Stepper<MemSpace::device> expected_step(
        this->make_stepper_input(num_tracks));
    Stepper<MemSpace::device> step(this->make_stepper_input(num_tracks));
    auto primaries = this->make_primaries(num_primaries);
    auto expected_result = expected_step(make_span(primaries));
    step.async(make_span(primaries));
    EXPECT_TRUE(step.valid());
    step.wait();
    EXPECT_TRUE(step.valid());
    auto result = step.get();
    expect_stepper_eq(expected_result, result);
    EXPECT_FALSE(step.valid());

    for (int i = 0; i < 2; ++i)
    {
        expected_result = expected_step();
        step.async();
        EXPECT_TRUE(step.valid());
        EXPECT_NO_THROW(static_cast<void>(step.ready()));
        step.wait();
        EXPECT_TRUE(step.valid());

        result = step.get();
        expect_stepper_eq(expected_result, result);
        EXPECT_FALSE(step.valid());
    }
}

TEST_F(SimpleComptonTest, reseed)
{
    constexpr auto M = MemSpace::host;
    size_type num_primaries = 1;
    size_type num_tracks = 1;

    Stepper<M> step(this->make_stepper_input(num_tracks));
    auto const primaries = this->make_primaries(num_primaries);
    auto const& params_ref = this->core()->ref<M>();
    auto const& state_ref
        = dynamic_cast<CoreState<M> const&>(step.state()).ref();
    SimTrackView sim{params_ref.sim, state_ref.sim, TrackSlotId{0}};
    RngEngine engine{params_ref.rng, state_ref.rng, TrackSlotId{0}};

    // First step: save next random number
    step.reseed(UniqueEventId{123});
    step(make_span(primaries));
    EXPECT_EQ(TrackStatus::alive, sim.status());
    EXPECT_EQ(TrackId{0}, sim.track_id());
    auto orig_next_random = engine();
    sim.status(TrackStatus::inactive);
    step();

    // Next event should be a different random number and zero track ID
    step.reseed(UniqueEventId{3456});
    step(make_span(primaries));
    EXPECT_EQ(TrackStatus::alive, sim.status());
    EXPECT_EQ(TrackId{0}, sim.track_id());
    EXPECT_NE(orig_next_random, engine());
    sim.status(TrackStatus::inactive);
    step();

    // Original event should have the same RNG sequence
    step.reseed(UniqueEventId{123});
    step(make_span(primaries));
    EXPECT_EQ(TrackStatus::alive, sim.status());
    EXPECT_EQ(TrackId{0}, sim.track_id());
    EXPECT_EQ(orig_next_random, engine());
}

TEST_F(SimpleComptonTest, kill_active)
{
    constexpr auto M = MemSpace::host;
    size_type num_primaries = 2;
    size_type num_tracks = 8;

    Stepper<M> step(this->make_stepper_input(num_tracks));
    auto const primaries = this->make_primaries(num_primaries);
    auto counters = step(make_span(primaries));
    EXPECT_EQ(2, counters.alive);

    // Fill time with something simple to make result more reproducible
    {
        auto const& state_ref
            = dynamic_cast<CoreState<M> const&>(step.state()).ref();
        auto time = state_ref.sim.time[AllItems<real_type>{}];
        std::fill(time.begin(), time.end(), 0.25);
    }

    ScopedLogStorer scoped_log{&celeritas::self_logger()};
    step.kill_active();
    counters = step();
    EXPECT_EQ(0, counters.alive);
    static char const* const expected_log_messages[] = {
        "Killing 2 active tracks",
        R"(Killing track {"geo":{"dir":[1.0,0.0,0.0],"is_on_boundary":true,"is_outside":false,"pos":[[-5.0,0.0,0.0],"cm"],"volume_id":"inner"},"mat":"Al","particle":{"energy":[100.0,"MeV"],"particle_id":"gamma"},"sim":{"event_id":0,"num_steps":1,"parent_id":null,"post_step_action":"tracking-cut","primary_id":null,"status":"errored","step_length":[17.0,"cm"],"time":[0.25,"s"],"track_id":0},"thread_id":6,"track_slot_id":6}: lost 100 MeV)",
        R"(Killing track {"geo":{"dir":[1.0,0.0,0.0],"is_on_boundary":true,"is_outside":false,"pos":[[-5.0,0.0,0.0],"cm"],"volume_id":"inner"},"mat":"Al","particle":{"energy":[100.0,"MeV"],"particle_id":"gamma"},"sim":{"event_id":0,"num_steps":1,"parent_id":null,"post_step_action":"tracking-cut","primary_id":null,"status":"errored","step_length":[17.0,"cm"],"time":[0.25,"s"],"track_id":1},"thread_id":7,"track_slot_id":7}: lost 100 MeV)",
    };
    if (CELERITAS_UNITS == CELERITAS_UNITS_CGS && CELERITAS_USE_GEANT4)
    {
        // Messages match only when importing Geant4 volumes
        EXPECT_VEC_EQ(expected_log_messages, scoped_log.messages())
            << scoped_log;
    }
    static char const* const expected_log_levels[]
        = {"error", "error", "error"};
    EXPECT_VEC_EQ(expected_log_levels, scoped_log.levels());
}

TEST_F(SimpleComptonTest, max_steps)
{
    max_steps_ = 2;

    size_type num_primaries = 2;
    size_type num_tracks = 64;

    Stepper<MemSpace::host> step(this->make_stepper_input(num_tracks));

    ScopedLogStorer scoped_log{&celeritas::self_logger(), LogLevel::debug};
    auto result = this->run(step, num_primaries);

    static char const* const expected_log_levels[]
        = {"debug", "debug", "debug", "debug"};
    EXPECT_VEC_EQ(expected_log_levels, scoped_log.levels());
    ASSERT_EQ(4, scoped_log.messages().size());
    EXPECT_EQ("Track {62} exceeded maximum step count",
              scoped_log.messages()[0]);
    EXPECT_TRUE(scoped_log.messages()[2].find("\"num_steps\":2")
                != std::string::npos);
}

//---------------------------------------------------------------------------//

TEST_F(StepperOrderTest, setup)
{
    auto result = this->check_setup();
    static char const* const expected_processes[] = {"Compton scattering"};
    EXPECT_VEC_EQ(expected_processes, result.processes);
    static char const* const expected_actions[] = {
        "extend-from-primaries",
        "initialize-tracks",
        "dummy-user_start",
        "pre-step",
        "dummy-user_pre",
        "along-step-neutral",
        "physics-discrete-select",
        "scat-klein-nishina",
        "geo-boundary",
        "tracking-cut",
        "dummy-user_post",
        "extend-from-secondaries",
    };
    EXPECT_VEC_EQ(expected_actions, result.actions);
}

TEST_F(StepperOrderTest, warm_up)
{
    Stepper<MemSpace::host> step(this->make_stepper_input(32));
    auto const& dumstate = this->get_dummy_state(step.state());

    EXPECT_EQ(0, dumstate.action_order.size());
    step.warm_up();
    EXPECT_EQ(0, step.state().sync_get_counters().num_active);
    EXPECT_EQ(0, step.state().sync_get_counters().num_alive);

    static char const* const expected_action_order[]
        = {"user_start", "user_pre", "user_post"};
    EXPECT_VEC_EQ(expected_action_order, dumstate.action_order);
}

TEST_F(StepperOrderTest, step)
{
    constexpr auto M = MemSpace::host;
    size_type num_primaries = 32;
    size_type num_tracks = 64;

    Stepper<M> step(this->make_stepper_input(num_tracks));
    auto const& state = this->get_dummy_state(step.state());

    EXPECT_EQ(0, state.action_order.size());
    EXPECT_EQ(M, state.memspace);
    EXPECT_EQ(StreamId{0}, state.stream_id);
    EXPECT_EQ(num_tracks, state.size);

    // Initialize some primaries and take a step
    auto primaries = this->make_primaries(num_primaries);
    step(make_span(primaries));

    static char const* const expected_action_order[]
        = {"user_start", "user_pre", "user_post"};
    EXPECT_VEC_EQ(expected_action_order, state.action_order);
}

//---------------------------------------------------------------------------//

TEST_F(BadGeometryTest, no_volume_host)
{
    auto scoped_log = this->run_one_failure<MemSpace::host>({-5, 0, 0});

    static char const* const expected_log_messages[] = {
        R"(Failed to initialize geometry state: could not find associated volume in universe 0 at local position {-5, 0, 0})",
        R"(Killing track {"geo":{"dir":[1.0,0.0,0.0],"is_on_boundary":false,"is_outside":true,"pos":[[-5.0,0.0,0.0],"cm"]},"mat":null,"particle":{"energy":[100.0,"MeV"],"particle_id":"gamma"},"sim":{"event_id":0,"num_steps":0,"parent_id":null,"post_step_action":"tracking-cut","primary_id":null,"status":"errored","step_length":[0.0,"cm"],"time":[0.0,"s"],"track_id":0},"thread_id":0,"track_slot_id":0}: depositing 100 MeV)",
    };
    if (CELERITAS_UNITS == CELERITAS_UNITS_CGS
        && CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        EXPECT_VEC_EQ(expected_log_messages, scoped_log.messages());
    }

    static char const* const expected_log_levels[] = {"error", "error"};
    EXPECT_VEC_EQ(expected_log_levels, scoped_log.levels());
}

TEST_F(BadGeometryTest, no_material_host)
{
    auto scoped_log = this->run_one_failure<MemSpace::host>({5, 0, 0});

    static char const* const expected_log_messages[] = {
        "Track 0 started in an unknown material",
        R"(Killing track {"geo":{"dir":[1.0,0.0,0.0],"is_on_boundary":false,"is_outside":false,"pos":[[5.0,0.0,0.0],"cm"],"volume_id":"[missing material]@world"},"mat":null,"particle":{"energy":[100.0,"MeV"],"particle_id":"gamma"},"sim":{"event_id":0,"num_steps":0,"parent_id":null,"post_step_action":"tracking-cut","primary_id":null,"status":"errored","step_length":[0.0,"cm"],"time":[0.0,"s"],"track_id":0},"thread_id":0,"track_slot_id":0}: lost 100 MeV)",
    };

    if (CELERITAS_UNITS == CELERITAS_UNITS_CGS)
    {
        EXPECT_VEC_EQ(expected_log_messages, scoped_log.messages());
    }
    EXPECT_EQ(2, scoped_log.levels().size());
}

TEST_F(BadGeometryTest, no_new_volume_host)
{
    auto scoped_log = this->run_one_failure<MemSpace::host>({-6.001, 0, 0});

    static char const* const expected_log_messages[] = {
        R"(track failed to cross local surface 2 in universe 0 at local position {-6, 0, 0} along local direction {1, 0, 0})",
        R"(Killing track {"geo":{"dir":[1.0,0.0,0.0],"is_on_boundary":true,"is_outside":true,"pos":[[-6.0,0.0,0.0],"cm"]},"mat":null,"particle":{"energy":[100.0,"MeV"],"particle_id":"gamma"},"sim":{"event_id":0,"num_steps":1,"parent_id":null,"post_step_action":"tracking-cut","primary_id":null,"status":"errored","step_length":[0.001000,"cm"],"time":[3.336e-14,"s"],"track_id":0},"thread_id":0,"track_slot_id":0}: depositing 100 MeV)",
    };

    if (CELERITAS_UNITS == CELERITAS_UNITS_CGS
        && CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        EXPECT_VEC_EQ(expected_log_messages, scoped_log.messages())
            << scoped_log;
    }
    static char const* const expected_log_levels[] = {"error", "error"};
    EXPECT_VEC_EQ(expected_log_levels, scoped_log.levels());
}

TEST_F(BadGeometryTest, start_outside_host)
{
    auto scoped_log = this->run_one_failure<MemSpace::host>({20, 0, 0});

    static char const* const expected_log_messages[] = {
        "Track 0 started outside the geometry",
        R"(Killing track {"geo":{"dir":[1.0,0.0,0.0],"is_on_boundary":false,"is_outside":true,"pos":[[20.0,0.0,0.0],"cm"]},"mat":null,"particle":{"energy":[100.0,"MeV"],"particle_id":"gamma"},"sim":{"event_id":0,"num_steps":0,"parent_id":null,"post_step_action":"tracking-cut","primary_id":null,"status":"errored","step_length":[0.0,"cm"],"time":[0.0,"s"],"track_id":0},"thread_id":0,"track_slot_id":0}: depositing 100 MeV)",
    };

    if (CELERITAS_UNITS == CELERITAS_UNITS_CGS)
    {
        EXPECT_VEC_EQ(expected_log_messages, scoped_log.messages());
    }
    EXPECT_EQ(2, scoped_log.levels().size());
}

TEST_F(BadGeometryTest, TEST_IF_CELER_DEVICE(no_volume_device))
{
    this->run_one_failure<MemSpace::device>({-5, 0, 0});
}

TEST_F(BadGeometryTest, TEST_IF_CELER_DEVICE(no_material_device))
{
    this->run_one_failure<MemSpace::device>({5, 0, 0});
}

TEST_F(BadGeometryTest, TEST_IF_CELER_DEVICE(no_new_volume_device))
{
    this->run_one_failure<MemSpace::device>({-6.001, 0, 0});
}

TEST_F(BadGeometryTest, TEST_IF_CELER_DEVICE(start_outside_device))
{
    this->run_one_failure<MemSpace::device>({20, 0, 0});
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
