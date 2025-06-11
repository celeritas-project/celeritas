//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/TrackInit.test.cc
//---------------------------------------------------------------------------//
#include <algorithm>
#include <initializer_list>
#include <numeric>
#include <vector>

#include "corecel/OpaqueIdUtils.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"
#include "corecel/data/CollectionStateStore.hh"
#include "corecel/data/Ref.hh"
#include "corecel/io/LogContextException.hh"
#include "corecel/sys/ActionRegistry.hh"
#include "celeritas/SimpleTestBase.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/CoreTrackData.hh"
#include "celeritas/track/ExtendFromPrimariesAction.hh"
#include "celeritas/track/ExtendFromSecondariesAction.hh"
#include "celeritas/track/InitializeTracksAction.hh"

#include "MockInteractAction.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// TEST RESULT
//---------------------------------------------------------------------------//

struct RunResult
{
    std::vector<int> track_ids;
    std::vector<int> parent_ids;
    std::vector<int> init_ids;
    std::vector<int> geo_parent_ids;
    std::vector<int> vacancies;

    template<MemSpace M>
    static RunResult from_state(CoreState<M>&);

    // Print code for the expected attributes
    void print_expected() const;
};

//---------------------------------------------------------------------------//

template<MemSpace M>
RunResult RunResult::from_state(CoreState<M>& state)
{
    RunResult result;

    // Copy track initializer data to host
    HostVal<TrackInitStateData> data;
    data = state.ref().init;

    // Store the IDs of the vacant track slots
    for (auto tid : range(TrackSlotId{state.counters().num_vacancies}))
    {
        result.vacancies.push_back(id_to_int(data.vacancies[tid]));
    }

    // Store the track IDs of the initializers
    for (auto init_id :
         range(ItemId<TrackInitializer>{state.counters().num_initializers}))
    {
        auto const& init = data.initializers[init_id];
        result.init_ids.push_back(id_to_int(init.sim.track_id));
        result.geo_parent_ids.push_back(id_to_int(init.geo.parent));
    }

    // Copy sim state to host
    // NOTE: if states have not been initialized on device, these IDs may also
    // be uninitialized
    HostVal<SimStateData> sim;
    sim = state.ref().sim;

    // Store the track IDs and parent IDs
    for (auto tid : range(TrackSlotId{sim.size()}))
    {
        result.track_ids.push_back(id_to_int(sim.track_ids[tid]));
        result.parent_ids.push_back(id_to_int(sim.parent_ids[tid]));
    }

    return result;
}

// Print code for the expected attributes
void RunResult::print_expected() const
{
    cout << "/*** ADD THE FOLLOWING UNIT TEST CODE ***/\n"
            "static int const expected_track_ids[] = "
         << repr(this->track_ids) << ";\n"
         << "EXPECT_VEC_EQ(expected_track_ids, result.track_ids);\n"
            "static int const expected_parent_ids[] = "
         << repr(this->parent_ids) << ";\n"
         << "EXPECT_VEC_EQ(expected_parent_ids, result.parent_ids);\n"
            "static int const expected_init_ids[] = "
         << repr(this->init_ids) << ";\n"
         << "EXPECT_VEC_EQ(expected_init_ids, result.init_ids);\n"
            "static int const expected_geo_parent_ids[] = "
         << repr(this->geo_parent_ids) << ";\n"
         << "EXPECT_VEC_EQ(expected_geo_parent_ids, result.geo_parent_ids);\n"
            "static int const expected_vacancies[] = "
         << repr(this->vacancies) << ";\n"
         << "EXPECT_VEC_EQ(expected_vacancies, result.vacancies);\n"
            "/*** END CODE ***/\n";
}

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class TrackInitTestBase : public SimpleTestBase
{
  protected:
    //! Create primary particles
    std::vector<Primary> make_primaries(size_type num_primaries) const
    {
        std::vector<Primary> result;
        for (unsigned int i = 0; i < num_primaries; ++i)
        {
            Primary p;
            p.particle_id = ParticleId{0};
            p.energy = units::MevEnergy(1 + i);
            p.position = {0, 0, 0};
            p.direction = {0, 0, 1};
            p.time = 0;
            p.event_id = EventId{0};
            result.push_back(p);
        }
        return result;
    }

    // Reserve extra space for secondaries
    real_type secondary_stack_factor() const override { return 8; }
};

//---------------------------------------------------------------------------//

template<class T>
class TrackInitTest : public TrackInitTestBase
{
  public:
    // Memspace for this class instance
    static constexpr MemSpace M = T::value;

    //! Create mutable state data
    void build_states(size_type num_tracks)
    {
        if constexpr (M == MemSpace::device)
        {
            device().create_streams(1);
        }
        // Allocate state data
        state_ = std::make_unique<CoreState<M>>(
            *this->core(), StreamId{0}, num_tracks);
    }

    CoreState<M>& state()
    {
        CELER_ENSURE(state_);
        return *state_;
    }

    void extend_from_primaries(Span<Primary const> primaries)
    {
        CELER_EXPECT(state_);
        this->insert_primaries(*state_, primaries);
        this->primaries_action()->step(*this->core(), *state_);
    }

    std::shared_ptr<CoreStepActionInterface const> pre_step_action() const
    {
        auto aid = this->action_reg()->find_action("pre-step");
        CELER_ASSERT(aid);
        return std::dynamic_pointer_cast<CoreStepActionInterface const>(
            this->action_reg()->action(aid));
    }
    void init_tracks()
    {
        // Initialize tracks
        InitializeTracksAction{ActionId{0}}.step(*this->core(), this->state());

        // Reset physics state before interacting
        this->pre_step_action()->step(*this->core(), *state_);
    }

  private:
    std::unique_ptr<CoreState<M>> state_;
};

using HostType = std::integral_constant<MemSpace, MemSpace::host>;
using DeviceType = std::integral_constant<MemSpace, MemSpace::device>;

#if CELER_USE_DEVICE
using MemspaceTypes = ::testing::Types<HostType, DeviceType>;
#else
using MemspaceTypes = ::testing::Types<HostType>;
#endif

struct MemspaceTypeString
{
    template<class U>
    static std::string GetName(int)
    {
        return celeritas::to_cstring(U::value);
    }
};

TYPED_TEST_SUITE(TrackInitTest, MemspaceTypes, MemspaceTypeString);

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

//! Test that we can add more primaries than the first allocation
TYPED_TEST(TrackInitTest, add_more_primaries)
{
    this->build_states(16);
    EXPECT_EQ(0, this->state().counters().num_initializers);

    auto primaries = this->make_primaries(22);
    this->extend_from_primaries(make_span(primaries));
    EXPECT_EQ(22, this->state().counters().num_initializers);

    primaries = this->make_primaries(32);
    this->extend_from_primaries(make_span(primaries));
    EXPECT_EQ(54, this->state().counters().num_initializers);
}

//! Test that we can add more primaries than the first allocation
TYPED_TEST(TrackInitTest, extend_primaries)
{
    this->build_states(8);

    {
        // Don't initialize
        auto primaries = this->make_primaries(2);
        this->insert_primaries(this->state(), make_span(primaries));
        RunResult::from_state(this->state());

        EXPECT_EQ(0, this->state().counters().num_initializers);
    }
    {
        // Now initialize after adding
        auto primaries = this->make_primaries(4);
        EXPECT_THROW(this->extend_from_primaries(make_span(primaries)),
                     RuntimeError);
        if (false)
        {
            // NOTE: feature is not yet implemented; re-enable when
            // EXPECT_THROW is removed
            this->init_tracks();
            auto result = RunResult::from_state(this->state());
            static int const expected_track_ids[] = {-1, -1, 0, 1, 0, 1, 2, 3};
            EXPECT_VEC_EQ(expected_track_ids, result.track_ids);
        }
    }
}

TYPED_TEST(TrackInitTest, run)
{
    size_type const num_primaries = 12;
    size_type const num_tracks = 10;

    this->build_states(num_tracks);

    // Check that all of the track slots were marked as empty
    {
        auto result = RunResult::from_state(this->state());
        static int const expected_vacancies[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        EXPECT_VEC_EQ(expected_vacancies, result.vacancies);
    }

    // Create track initializers on device from primary particles
    auto primaries = this->make_primaries(num_primaries);
    this->extend_from_primaries(make_span(primaries));

    // Check the track IDs of the track initializers created from primaries
    {
        auto result = RunResult::from_state(this->state());
        static int const expected_track_ids[]
            = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        EXPECT_VEC_EQ(expected_track_ids, result.init_ids);
    }

    // Initialize the primary tracks on device
    this->init_tracks();

    // Check the track IDs and parent IDs of the initialized tracks
    {
        auto result = RunResult::from_state(this->state());
        static int const expected_track_ids[]
            = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        EXPECT_VEC_EQ(expected_track_ids, result.track_ids);

        // All primary particles, so no parent
        static int const expected_parent_ids[]
            = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
        EXPECT_VEC_EQ(expected_parent_ids, result.parent_ids);
    }

    auto interact = [] {
        // Number of secondaries to produce for each track and whether the
        // track survives the interaction
        std::vector<size_type> const alloc = {1, 1, 0, 0, 1, 1, 0, 0, 2, 1};
        std::vector<bool> const alive = {
            false, true, false, true, false, true, false, true, false, false};
        return MockInteractAction{ActionId{1}, alloc, alive};
    }();
    interact.step(*this->core(), this->state());

    // Launch a kernel to create track initializers from secondaries
    ExtendFromSecondariesAction{ActionId{2}}.step(*this->core(), this->state());

    {
        // Check the vacancies
        auto result = RunResult::from_state(this->state());
        static int const expected_vacancies[] = {2, 6};
        EXPECT_VEC_EQ(expected_vacancies, result.vacancies);

        // Check the parent IDs for copying the geometry state
        static int const expected_geo_parent_ids[] = {-1, -1, -1, 5, 8};
        EXPECT_VEC_EQ(expected_geo_parent_ids, result.geo_parent_ids);

        // Check the track IDs of the track initializers created from
        // secondaries. Because IDs are not calculated deterministically and we
        // don't know which IDs were used for the immediately-initialized
        // secondaries and which were used for the track initializers, just
        // check that there is the correct number and they are in the correct
        // range.
        EXPECT_TRUE(std::all_of(std::begin(result.init_ids),
                                std::end(result.init_ids),
                                [](int id) { return id >= 0 && id <= 18; }));
        ASSERT_EQ(5, result.init_ids.size());

        // First two initializers are from primaries
        EXPECT_EQ(0, result.init_ids[0]);
        EXPECT_EQ(1, result.init_ids[1]);
    }

    // Initialize secondaries on device
    this->init_tracks();

    // Check the track IDs and parent IDs of the initialized tracks
    {
        auto result = RunResult::from_state(this->state());
        EXPECT_TRUE(std::all_of(std::begin(result.track_ids),
                                std::end(result.track_ids),
                                [](int id) { return id >= 0 && id <= 18; }));

        // Tracks that were not killed should have the same ID
        EXPECT_EQ(3, result.track_ids[1]);
        EXPECT_EQ(5, result.track_ids[3]);
        EXPECT_EQ(7, result.track_ids[5]);
        EXPECT_EQ(9, result.track_ids[7]);

        // Two tracks should have the same parent ID = 10
        int expected_parent_ids[] = {2, -1, 7, -1, 6, -1, 10, -1, 10, 11};
        std::sort(std::begin(result.parent_ids), std::end(result.parent_ids));
        std::sort(std::begin(expected_parent_ids),
                  std::end(expected_parent_ids));
        EXPECT_VEC_EQ(expected_parent_ids, result.parent_ids);

        // Check the parent IDs for copying the geometry state
        static int const expected_geo_parent_ids[] = {-1, -1, -1};
        EXPECT_VEC_EQ(expected_geo_parent_ids, result.geo_parent_ids);
    }
}

TYPED_TEST(TrackInitTest, primaries)
{
    size_type num_sets = 4;
    size_type num_primaries = 16;
    size_type num_tracks = 16;

    this->build_states(num_tracks);

    this->init_tracks();

    // Kill half the tracks in each interaction and don't produce secondaries
    auto interact = [&] {
        std::vector<size_type> alloc(num_tracks, 0);
        std::vector<bool> alive(num_tracks);
        for (size_type i = 0; i < num_tracks; ++i)
        {
            alive[i] = i % 2;
        }
        return MockInteractAction{ActionId{1}, alloc, alive};
    }();

    ExtendFromSecondariesAction extend_from_secondaries{ActionId{2}};

    for (size_type i = 0; i < num_sets; ++i)
    {
        // Create track initializers on device from primary particles
        auto primaries = this->make_primaries(num_primaries);
        this->extend_from_primaries(make_span(primaries));

        // Initialize tracks on device
        this->init_tracks();

        // Launch kernel that will kill half the tracks
        interact.step(*this->core(), this->state());

        // Find vacancies and create track initializers from secondaries
        extend_from_secondaries.step(*this->core(), this->state());
        EXPECT_EQ(i * num_tracks / 2,
                  this->state().counters().num_initializers);
        EXPECT_EQ(num_tracks / 2, this->state().counters().num_vacancies);
    }

    // Check the results
    static int const expected_track_ids[] = {
        56,
        1,
        57,
        3,
        58,
        5,
        59,
        7,
        60,
        9,
        61,
        11,
        62,
        13,
        63,
        15,
    };
    static int const expected_parent_ids[]
        = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
    static int const expected_vacancies[] = {0, 2, 4, 6, 8, 10, 12, 14};
    static int const expected_init_ids[] = {
        16, 17, 18, 19, 20, 21, 22, 23, 32, 33, 34, 35,
        36, 37, 38, 39, 48, 49, 50, 51, 52, 53, 54, 55,
    };
    auto result = RunResult::from_state(this->state());
    EXPECT_VEC_EQ(expected_track_ids, result.track_ids);
    EXPECT_VEC_EQ(expected_parent_ids, result.parent_ids);
    EXPECT_VEC_EQ(expected_vacancies, result.vacancies);
    EXPECT_VEC_EQ(expected_init_ids, result.init_ids);
}

TYPED_TEST(TrackInitTest, extend_from_secondaries)
{
    // Basic setup
    size_type num_primaries = 8;
    size_type num_tracks = 8;

    std::vector<bool> const alive
        = {true, false, false, true, true, false, false, true};

    this->build_states(num_tracks);

    // Create actions
    std::vector<std::shared_ptr<CoreStepActionInterface const>> actions = {
        std::make_shared<InitializeTracksAction>(ActionId{0}),
        this->pre_step_action(),
        std::make_shared<MockInteractAction>(
            ActionId{1}, std::vector<size_type>{1, 1, 2, 0, 0, 0, 0, 0}, alive),
        std::make_shared<ExtendFromSecondariesAction>(ActionId{2})};

    // Create track initializers on device from primary particles
    auto primaries = this->make_primaries(num_primaries);
    this->extend_from_primaries(make_span(primaries));
    EXPECT_EQ(num_primaries, this->state().counters().num_initializers);

    auto apply_actions = [&actions, this] {
        for (auto const& ea_interface : actions)
        {
            ea_interface->step(*this->core(), this->state());
        }
    };

    for (int i : range(4))
    {
        CELER_TRY_HANDLE(apply_actions(),
                         LogContextException{this->output_reg().get()});
        auto result = RunResult::from_state(this->state());

        // Slots 5 and 6 are always vacant because these tracks are killed with
        // no secondaries
        static int const expected_vacancies[] = {5u, 6u};
        EXPECT_VEC_EQ(expected_vacancies, result.vacancies);

        static int const expected_geo_parent_ids[] = {0, 2};
        EXPECT_VEC_EQ(expected_geo_parent_ids, result.geo_parent_ids);

        // init ids may not be deterministic, but can guarantee they are in the
        // range 8<=x<=12 as we create 4 tracks per iteration, 2 in reused
        // slots from their parent, 2 as new inits
        EXPECT_EQ(2, result.init_ids.size());
        EXPECT_TRUE(std::all_of(
            std::begin(result.init_ids),
            std::end(result.init_ids),
            [i](int id) { return id >= 8 + i * 4 && id <= 11 + i * 4; }));

        // Track ids may not be deterministic, so only validate size and
        // range. (Remember that we create 4 new tracks per iteration, with 2
        // slots reused
        EXPECT_EQ(num_tracks, result.track_ids.size());
        EXPECT_TRUE(std::all_of(std::begin(result.track_ids),
                                std::end(result.track_ids),
                                [&](int id) {
                                    return id >= 0
                                           && id < static_cast<int>(num_tracks)
                                                       + (i + 1) * 4;
                                }));

        // Parent ids may not be deterministic, but all non-killed tracks are
        // guaranteed to be primaries at every iteration. At end of first
        // iteration, will still have some primary ids as these are not cleared
        // until the next iteration
        for (size_type pidx : range(num_tracks))
        {
            EXPECT_TRUE(((alive[pidx] == 1) ? result.parent_ids[pidx] == -1
                                            : result.parent_ids[pidx] >= -1))
                << "iteration " << i;
        }
    }
}  // namespace test

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
