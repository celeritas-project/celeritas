//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/TrackInit.test.cc
//---------------------------------------------------------------------------//
#include "TrackInit.test.hh"

#include <algorithm>
#include <numeric>
#include <vector>

#include "corecel/cont/Span.hh"
#include "corecel/data/CollectionMirror.hh"
#include "celeritas/SimpleTestBase.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreTrackData.hh"
#include "celeritas/track/TrackInitParams.hh"
#include "celeritas/track/TrackInitUtils.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
using TrackInitDeviceValue
    = TrackInitStateData<Ownership::value, MemSpace::device>;

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//

ITTestInput::ITTestInput(std::vector<size_type>& host_alloc_size,
                         std::vector<char>& host_alive)
    : alloc_size(host_alloc_size.size()), alive(host_alive.size())
{
    CELER_EXPECT(host_alloc_size.size() == host_alive.size());
    alloc_size.copy_to_device(make_span(host_alloc_size));
    alive.copy_to_device(make_span(host_alive));
}

ITTestInputData ITTestInput::device_ref()
{
    ITTestInputData result;
    result.alloc_size = alloc_size.device_ref();
    result.alive = alive.device_ref();
    return result;
}

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

#define TrackInitTest TEST_IF_CELER_DEVICE(TrackInitTest)

class TrackInitTest : public SimpleTestBase
{
  protected:
    void SetUp() override
    {
        core_data.params = this->core()->device_ref();
        CELER_ENSURE(core_data.params);
    }

    //! Create primary particles
    std::vector<Primary> generate_primaries(size_type num_primaries)
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

    //! Create mutable state data
    void build_states(size_type num_tracks)
    {
        CELER_EXPECT(core_data.params);

        // Allocate state data
        resize(&device_states, this->core()->host_ref(), num_tracks);
        core_data.states = device_states;

        CELER_ENSURE(core_data.states);
    }

    //! Copy results to host
    ITTestOutput get_result(CoreStateDeviceRef& states)
    {
        CELER_EXPECT(states);

        ITTestOutput result;

        // Copy track initializer data to host
        HostVal<TrackInitStateData> data;
        data = states.init;

        // Store the IDs of the vacant track slots
        auto const vacancies = data.vacancies.data();
        result.vacancies = {vacancies.begin(), vacancies.end()};

        // Store the track IDs of the initializers
        for (auto const& init : data.initializers.data())
        {
            result.init_ids.push_back(init.sim.track_id.get());
        }

        // Copy sim states to host
        StateCollection<SimTrackState, Ownership::value, MemSpace::host> sim(
            states.sim.state);

        // Store the track IDs and parent IDs
        for (auto tid : range(ThreadId{sim.size()}))
        {
            result.track_ids.push_back(sim[tid].track_id.unchecked_get());
            result.parent_ids.push_back(sim[tid].parent_id.unchecked_get());
        }

        return result;
    }

    CoreStateData<Ownership::value, MemSpace::device> device_states;
    CoreDeviceRef core_data;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(TrackInitTest, run)
{
    const size_type num_primaries = 12;
    const size_type num_tracks = 10;

    build_states(num_tracks);

    // Check that all of the track slots were marked as empty
    {
        auto result = get_result(core_data.states);
        static unsigned int const expected_vacancies[]
            = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        EXPECT_VEC_EQ(expected_vacancies, result.vacancies);
    }

    // Create track initializers on device from primary particles
    auto primaries = generate_primaries(num_primaries);
    extend_from_primaries(core_data, make_span(primaries));

    // Check the track IDs of the track initializers created from primaries
    {
        auto result = get_result(core_data.states);
        static unsigned int const expected_track_ids[]
            = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        EXPECT_VEC_EQ(expected_track_ids, result.init_ids);
    }

    // Initialize the primary tracks on device
    initialize_tracks(core_data);

    // Check the track IDs and parent IDs of the initialized tracks
    {
        auto result = get_result(core_data.states);
        static unsigned int const expected_track_ids[]
            = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        EXPECT_VEC_EQ(expected_track_ids, result.track_ids);

        // All primary particles, so no parent
        static int const expected_parent_ids[]
            = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
        EXPECT_VEC_EQ(expected_parent_ids, result.parent_ids);
    }

    // Allocate input device data (number of secondaries to produce for each
    // track and whether the track survives the interaction)
    std::vector<size_type> alloc = {1, 1, 0, 0, 1, 1, 0, 0, 2, 1};
    std::vector<char> alive = {0, 1, 0, 1, 0, 1, 0, 1, 0, 0};
    ITTestInput input(alloc, alive);

    // Launch kernel to process interactions
    interact(core_data.states, input.device_ref());

    // Launch a kernel to create track initializers from secondaries
    extend_from_secondaries(core_data);

    // Check the vacancies
    {
        auto result = get_result(core_data.states);
        static unsigned int const expected_vacancies[] = {2, 6};
        EXPECT_VEC_EQ(expected_vacancies, result.vacancies);
    }

    // Check the track IDs of the track initializers created from secondaries.
    // Because IDs are not calculated deterministically and we don't know which
    // IDs were used for the immediately-initialized secondaries and which were
    // used for the track initializers, just check that there is the correct
    // number and they are in the correct range.
    {
        auto result = get_result(core_data.states);
        EXPECT_TRUE(std::all_of(std::begin(result.init_ids),
                                std::end(result.init_ids),
                                [](unsigned int id) { return id <= 18; }));
        EXPECT_EQ(5, result.init_ids.size());

        // First two initializers are from primaries
        EXPECT_EQ(0, result.init_ids[0]);
        EXPECT_EQ(1, result.init_ids[1]);
    }

    // Initialize secondaries on device
    initialize_tracks(core_data);

    // Check the track IDs and parent IDs of the initialized tracks
    {
        auto result = get_result(core_data.states);
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

TEST_F(TrackInitTest, primaries)
{
    const size_type num_sets = 4;
    const size_type num_primaries = 16;
    const size_type num_tracks = 16;

    build_states(num_tracks);

    // Kill half the tracks in each interaction and don't produce secondaries
    std::vector<size_type> alloc(num_tracks, 0);
    std::vector<char> alive(num_tracks);
    for (size_type i = 0; i < num_tracks; ++i)
    {
        alive[i] = i % 2;
    }
    ITTestInput input(alloc, alive);

    for (size_type i = 0; i < num_sets; ++i)
    {
        // Create track initializers on device from primary particles
        auto primaries = generate_primaries(num_primaries);
        extend_from_primaries(core_data, make_span(primaries));

        // Initialize tracks on device
        initialize_tracks(core_data);

        // Launch kernel that will kill half the tracks
        interact(core_data.states, input.device_ref());

        // Find vacancies and create track initializers from secondaries
        extend_from_secondaries(core_data);
        EXPECT_EQ(i * num_tracks / 2,
                  core_data.states.init.initializers.size());
        EXPECT_EQ(num_tracks / 2, core_data.states.init.vacancies.size());
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
    auto result = get_result(core_data.states);
    EXPECT_VEC_EQ(expected_track_ids, result.track_ids);
    EXPECT_VEC_EQ(expected_parent_ids, result.parent_ids);
    EXPECT_VEC_EQ(expected_vacancies, result.vacancies);
    EXPECT_VEC_EQ(expected_init_ids, result.init_ids);
}

#define TrackInitSecondaryTest TEST_IF_CELER_DEVICE(TrackInitSecondaryTest)

class TrackInitSecondaryTest : public TrackInitTest
{
  protected:
    real_type secondary_stack_factor() const final { return 8; }
};

TEST_F(TrackInitSecondaryTest, secondaries)
{
    const size_type num_primaries = 512;
    const size_type num_tracks = 512;

    build_states(num_tracks);

    // Allocate input device data (number of secondaries to produce for each
    // track and whether the track survives the interaction)
    std::vector<size_type> alloc = {1, 1, 2, 0, 0, 0, 0, 0};
    std::vector<char> alive = {1, 0, 0, 1, 1, 0, 0, 1};
    size_type base_size = alive.size();
    for (size_type i = 0; i < num_tracks / base_size - 1; ++i)
    {
        alloc.insert(alloc.end(), alloc.begin(), alloc.begin() + base_size);
        alive.insert(alive.end(), alive.begin(), alive.begin() + base_size);
    }
    ITTestInput input(alloc, alive);

    // Create track initializers on device from primary particles
    auto primaries = generate_primaries(num_primaries);
    extend_from_primaries(core_data, make_span(primaries));
    EXPECT_EQ(num_primaries, core_data.states.init.initializers.size());

    const size_type num_iter = 16;
    for (CELER_MAYBE_UNUSED size_type i : range(num_iter))
    {
        // All queued initializers are converted to tracks
        initialize_tracks(core_data);
        ASSERT_EQ(0, core_data.states.init.initializers.size());

        // Launch kernel to process interactions
        interact(core_data.states, input.device_ref());

        // Launch a kernel to create track initializers from secondaries
        extend_from_secondaries(core_data);
        ASSERT_EQ(128, core_data.states.init.initializers.size())
            << "iteration " << i;
        ASSERT_EQ(128, core_data.states.init.vacancies.size());
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
