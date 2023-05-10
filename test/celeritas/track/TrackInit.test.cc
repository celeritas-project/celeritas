//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/TrackInit.test.cc
//---------------------------------------------------------------------------//
#include <algorithm>
#include <initializer_list>
#include <numeric>
#include <vector>

#include "corecel/cont/Span.hh"
#include "corecel/data/CollectionStateStore.hh"
#include "corecel/data/Ref.hh"
#include "corecel/io/LogContextException.hh"
#include "celeritas/SimpleTestBase.hh"
#include "celeritas/global/ActionRegistry.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreTrackData.hh"
#include "celeritas/track/ExtendFromSecondariesAction.hh"
#include "celeritas/track/InitializeTracksAction.hh"
#include "celeritas/track/TrackInitUtils.hh"

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
    std::vector<unsigned int> track_ids;
    std::vector<int> parent_ids;
    std::vector<unsigned int> init_ids;
    std::vector<size_type> vacancies;

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
    for (auto tid : range(TrackSlotId{data.scalars.num_vacancies}))
    {
        result.vacancies.push_back(data.vacancies[tid].unchecked_get());
    }

    // Store the track IDs of the initializers
    for (auto init_id :
         range(ItemId<TrackInitializer>{data.scalars.num_initializers}))
    {
        auto const& init = data.initializers[init_id];
        result.init_ids.push_back(init.sim.track_id.get());
    }

    // Copy sim state to host
    HostVal<SimStateData> sim;
    sim = state.ref().sim;

    // Store the track IDs and parent IDs
    for (auto tid : range(TrackSlotId{sim.size()}))
    {
        result.track_ids.push_back(sim.track_ids[tid].unchecked_get());
        result.parent_ids.push_back(sim.parent_ids[tid].unchecked_get());
    }

    return result;
}

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class TrackInitTest : public SimpleTestBase
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
            p.energy = units::MevEnergy{1. + i};
            p.position = {0, 0, 0};
            p.direction = {0, 0, 1};
            p.time = 0;
            p.event_id = EventId{0};
            p.track_id = TrackId{i};
            result.push_back(p);
        }
        return result;
    }

    // Reserve extra space for secondaries
    real_type secondary_stack_factor() const override { return 8; }
};

//---------------------------------------------------------------------------//

template<class T>
class TypedTrackInitTest : public TrackInitTest
{
  public:
    // Memspace for this class instance
    static constexpr MemSpace M = T::value;

    //! Create mutable state data
    void build_states(size_type num_tracks)
    {
        // Allocate state data
        state_ = std::make_unique<CoreState<M>>(
            *this->core(), StreamId{0}, num_tracks);
    }

    CoreState<M>& state()
    {
        CELER_ENSURE(state_);
        return *state_;
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

TYPED_TEST_SUITE(TypedTrackInitTest, MemspaceTypes, MemspaceTypeString);

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TYPED_TEST(TypedTrackInitTest, run)
{
    const size_type num_primaries = 12;
    const size_type num_tracks = 10;

    this->build_states(num_tracks);

    // Check that all of the track slots were marked as empty
    {
        auto result = RunResult::from_state(this->state());
        static unsigned int const expected_vacancies[]
            = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        EXPECT_VEC_EQ(expected_vacancies, result.vacancies);
    }

    // Create track initializers on device from primary particles
    auto primaries = this->make_primaries(num_primaries);
    extend_from_primaries(*this->core(), this->state(), make_span(primaries));

    // Check the track IDs of the track initializers created from primaries
    {
        auto result = RunResult::from_state(this->state());
        static unsigned int const expected_track_ids[]
            = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        EXPECT_VEC_EQ(expected_track_ids, result.init_ids);
    }

    // Initialize the primary tracks on device
    initialize_tracks(*this->core(), this->state());

    // Check the track IDs and parent IDs of the initialized tracks
    {
        auto result = RunResult::from_state(this->state());
        static unsigned int const expected_track_ids[]
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
        return MockInteractAction{ActionId{0}, alloc, alive};
    }();

    // Launch kernel to process interactions
    interact.execute(*this->core(), this->state());

    // Launch a kernel to create track initializers from secondaries
    extend_from_secondaries(*this->core(), this->state());

    // Check the vacancies
    {
        auto result = RunResult::from_state(this->state());
        static unsigned int const expected_vacancies[] = {2, 6};
        EXPECT_VEC_EQ(expected_vacancies, result.vacancies);
    }

    // Check the track IDs of the track initializers created from secondaries.
    // Because IDs are not calculated deterministically and we don't know which
    // IDs were used for the immediately-initialized secondaries and which were
    // used for the track initializers, just check that there is the correct
    // number and they are in the correct range.
    {
        auto result = RunResult::from_state(this->state());
        EXPECT_TRUE(std::all_of(std::begin(result.init_ids),
                                std::end(result.init_ids),
                                [](unsigned int id) { return id <= 18; }));
        EXPECT_EQ(5, result.init_ids.size());

        // First two initializers are from primaries
        EXPECT_EQ(0, result.init_ids[0]);
        EXPECT_EQ(1, result.init_ids[1]);
    }

    // Initialize secondaries on device
    initialize_tracks(*this->core(), this->state());

    // Check the track IDs and parent IDs of the initialized tracks
    {
        auto result = RunResult::from_state(this->state());
        EXPECT_TRUE(std::all_of(std::begin(result.track_ids),
                                std::end(result.track_ids),
                                [](unsigned int id) { return id <= 18; }));

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
    }
}

TYPED_TEST(TypedTrackInitTest, primaries)
{
    const size_type num_sets = 4;
    const size_type num_primaries = 16;
    const size_type num_tracks = 16;

    this->build_states(num_tracks);

    // Kill half the tracks in each interaction and don't produce secondaries
    auto interact = [] {
        std::vector<size_type> alloc(num_tracks, 0);
        std::vector<bool> alive(num_tracks);
        for (size_type i = 0; i < num_tracks; ++i)
        {
            alive[i] = i % 2;
        }
        return MockInteractAction{ActionId{0}, alloc, alive};
    }();

    for (size_type i = 0; i < num_sets; ++i)
    {
        // Create track initializers on device from primary particles
        auto primaries = this->make_primaries(num_primaries);
        extend_from_primaries(
            *this->core(), this->state(), make_span(primaries));

        // Initialize tracks on device
        initialize_tracks(*this->core(), this->state());

        // Launch kernel that will kill half the tracks
        interact.execute(*this->core(), this->state());

        // Find vacancies and create track initializers from secondaries
        extend_from_secondaries(*this->core(), this->state());
        auto& init = this->state().ref().init;
        EXPECT_EQ(i * num_tracks / 2, init.scalars.num_initializers);
        EXPECT_EQ(num_tracks / 2, init.scalars.num_vacancies);
    }

    // Check the results
    static unsigned int const expected_track_ids[] = {
        8u, 1u, 9u, 3u, 10u, 5u, 11u, 7u, 12u, 9u, 13u, 11u, 14u, 13u, 15u, 15u};
    static int const expected_parent_ids[]
        = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
    static unsigned int const expected_vacancies[]
        = {0u, 2u, 4u, 6u, 8u, 10u, 12u, 14u};
    static unsigned int const expected_init_ids[]
        = {0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 0u, 1u, 2u, 3u,
           4u, 5u, 6u, 7u, 0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u};
    auto result = RunResult::from_state(this->state());
    EXPECT_VEC_EQ(expected_track_ids, result.track_ids);
    EXPECT_VEC_EQ(expected_parent_ids, result.parent_ids);
    EXPECT_VEC_EQ(expected_vacancies, result.vacancies);
    EXPECT_VEC_EQ(expected_init_ids, result.init_ids);
}

TYPED_TEST(TypedTrackInitTest, secondaries)
{
    const size_type num_groups = 32;
    const size_type num_tracks = 8 * num_groups;
    this->build_states(num_tracks);

    auto interact = [] {
        size_type const nsec_inp[] = {1, 1, 2, 0, 0, 0, 0, 0};
        bool const alive_inp[]
            = {true, false, false, true, true, false, false, true};
        std::vector<size_type> nsec;
        std::vector<bool> alive;
        for (size_type i = 0; i < num_groups; ++i)
        {
            nsec.insert(nsec.end(), std::begin(nsec_inp), std::end(nsec_inp));
            alive.insert(
                alive.end(), std::begin(alive_inp), std::end(alive_inp));
        }
        return MockInteractAction{ActionId{0}, nsec, alive};
    }();

    // Create track initializers on device from primary particles
    const size_type num_primaries = num_tracks;
    auto primaries = this->make_primaries(num_primaries);
    extend_from_primaries(*this->core(), this->state(), make_span(primaries));
    EXPECT_EQ(num_primaries, this->state().ref().init.scalars.num_initializers);

    const size_type num_iter = 16;
    for ([[maybe_unused]] size_type i : range(num_iter))
    {
        SCOPED_TRACE(i);
        auto& init = this->state().ref().init;

        // All queued initializers are converted to tracks
        initialize_tracks(*this->core(), this->state());
        ASSERT_EQ(0, init.scalars.num_initializers);
        EXPECT_EQ(0, init.scalars.num_vacancies);

        // Launch kernel to process interactions
        interact.execute(*this->core(), this->state());

        // Launch a kernel to create track initializers from secondaries
        extend_from_secondaries(*this->core(), this->state());
        EXPECT_EQ(num_groups * 2, init.scalars.num_initializers);
        EXPECT_EQ(num_groups * 2, init.scalars.num_vacancies);

        // Number of secondaries *excludes* in-place secondaries: this is
        // really the number of initializers to be consumed
        EXPECT_EQ(num_groups * (1 + 0 + 1), init.scalars.num_secondaries);
        if (this->HasFailure())
        {
            FAIL() << "Aborting loop";
        }
    }
}

TYPED_TEST(TypedTrackInitTest, secondaries_action)
{
    // Basic setup
    const size_type num_primaries = 8;
    const size_type num_tracks = 8;

    std::vector<bool> const alive
        = {true, false, false, true, true, false, false, true};

    this->build_states(num_tracks);

    // Create actions
    std::vector<std::shared_ptr<ExplicitActionInterface>> actions = {
        std::make_shared<InitializeTracksAction>(ActionId{0}),
        std::make_shared<MockInteractAction>(
            ActionId{1}, std::vector<size_type>{1, 1, 2, 0, 0, 0, 0, 0}, alive),
        std::make_shared<ExtendFromSecondariesAction>(ActionId{2})};

    // Create track initializers on device from primary particles
    auto primaries = this->make_primaries(num_primaries);
    extend_from_primaries(*this->core(), this->state(), make_span(primaries));
    EXPECT_EQ(num_primaries, this->state().ref().init.scalars.num_initializers);

    auto apply_actions = [&actions, this] {
        for (const auto& ea_interface : actions)
        {
            ea_interface->execute(*this->core(), this->state());
        }
    };

    const size_type num_iter = 4;
    for ([[maybe_unused]] size_type i : range(num_iter))
    {
        CELER_TRY_HANDLE(apply_actions(),
                         LogContextException{this->output_reg().get()});
        auto result = RunResult::from_state(this->state());

        // Slots 5 and 6 are always vacant because these tracks are killed with
        // no secondaries
        static unsigned int const expected_vacancies[] = {5u, 6u};
        EXPECT_VEC_EQ(expected_vacancies, result.vacancies);

        // init ids may not be deterministic, but can guarantee they are in the
        // range 8<=x<=12 as we create 4 tracks per iteration, 2 in reused
        // slots from their parent, 2 as new inits
        EXPECT_EQ(2, result.init_ids.size());
        EXPECT_TRUE(std::all_of(std::begin(result.init_ids),
                                std::end(result.init_ids),
                                [i](unsigned int id) {
                                    return id >= 8 + i * 4 && id <= 11 + i * 4;
                                }));

        // Track ids may not be deterministic, so only validate size and
        // range. (Remember that we create 4 new tracks per iteration, with 2
        // slots reused
        EXPECT_EQ(num_tracks, result.track_ids.size());
        EXPECT_TRUE(std::all_of(
            std::begin(result.track_ids),
            std::end(result.track_ids),
            [i](unsigned int id) { return id < num_tracks + (i + 1) * 4; }));

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
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
