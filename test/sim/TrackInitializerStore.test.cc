//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file TrackInitializerStore.test.cc
//---------------------------------------------------------------------------//
#include "sim/TrackInitializerStore.hh"

#include <algorithm>
#include <numeric>
#include "celeritas_test.hh"
#include "base/CollectionStateStore.hh"
#include "geometry/GeoParams.hh"
#include "physics/base/ParticleParams.hh"
#include "physics/material/MaterialParams.hh"
#include "sim/TrackInterface.hh"
#include "sim/StateStore.hh"
#include "sim/TrackInitializerStore.hh"
#include "TrackInitializerStore.test.hh"

namespace celeritas_test
{
using namespace celeritas;

template<Ownership W, MemSpace M>
using SecondaryAllocatorData = celeritas::StackAllocatorData<Secondary, W, M>;

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//

ITTestInput::ITTestInput(std::vector<size_type>& host_alloc_size,
                         std::vector<char>&      host_alive)
    : alloc_size(host_alloc_size.size()), alive(host_alive.size())
{
    CELER_EXPECT(host_alloc_size.size() == host_alive.size());
    alloc_size.copy_to_device(make_span(host_alloc_size));
    alive.copy_to_device(make_span(host_alive));
}

ITTestInputPointers ITTestInput::device_pointers()
{
    ITTestInputPointers result;
    result.alloc_size = alloc_size.device_pointers();
    result.alive      = alive.device_pointers();
    return result;
}

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class TrackInitTest : public celeritas::Test
{
  protected:
    void SetUp() override
    {
        // Set up shared geometry data
        std::string test_file
            = celeritas::Test::test_data_path("geometry", "twoBoxes.gdml");
        geo_params = std::make_shared<GeoParams>(test_file.c_str());

        // Set up shared material data
        MaterialParams::Input mats;
        mats.elements  = {{{1, units::AmuMass{1.008}, "H"}}};
        mats.materials = {{{1e-5 * constants::na_avogadro,
                            100.0,
                            MatterState::gas,
                            {{ElementId{0}, 1.0}},
                            "H2"}}};
        auto material_params
            = std::make_shared<MaterialParams>(std::move(mats));

        auto particle_params = std::make_shared<ParticleParams>(
            ParticleParams::Input{{"gamma",
                                   pdg::gamma(),
                                   zero_quantity(),
                                   zero_quantity(),
                                   ParticleDef::stable_decay_constant()}});

        // Set the shared problem data
        params = ParamStore(geo_params, material_params, particle_params);
    }

    // Create primary particles
    std::vector<Primary> generate_primaries(size_type num_primaries)
    {
        std::vector<Primary> result;
        for (unsigned int i = 0; i < num_primaries; ++i)
        {
            result.push_back({ParticleId{0},
                              units::MevEnergy{1. + i},
                              {0., 0., 0.},
                              {0., 0., 1.},
                              EventId{0},
                              TrackId{i}});
        }
        return result;
    }

    std::shared_ptr<GeoParams> geo_params;
    ParamStore                 params;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(TrackInitTest, run)
{
    const size_type num_tracks = 10;
    const size_type capacity   = 100;

    // Create 12 primary particles
    std::vector<Primary> primaries = generate_primaries(12);

    // Allocate storage on device
    StateStore states({num_tracks, geo_params, 12345u});
    CollectionStateStore<SecondaryAllocatorData, MemSpace::device> secondaries(
        capacity);
    TrackInitializerStore track_init(num_tracks, capacity, primaries);

    // Check that all of the track slots were marked as empty
    ITTestOutput output, expected;
    output.vacancy   = vacancies_test(track_init.device_pointers());
    expected.vacancy = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    EXPECT_VEC_EQ(expected.vacancy, output.vacancy);

    // Create track initializers on device from primary particles
    track_init.extend_from_primaries();

    // Check the track IDs of the track initializers created from primaries
    output.initializer_id   = initializers_test(track_init.device_pointers());
    expected.initializer_id = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    EXPECT_VEC_EQ(expected.initializer_id, output.initializer_id);

    // Initialize the primary tracks on device
    track_init.initialize_tracks(&states, &params);

    // Check the IDs of the initialized tracks
    output.track_id   = tracks_test(states.device_pointers());
    expected.track_id = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    EXPECT_VEC_EQ(expected.track_id, output.track_id);

    // Allocate input device data (number of secondaries to produce for each
    // track and whether the track survives the interaction)
    std::vector<size_type> alloc = {1, 1, 0, 0, 1, 1, 0, 0, 1, 1};
    std::vector<char>      alive = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
    ITTestInput            input(alloc, alive);

    // Launch kernel to process interactions
    interact(
        states.device_pointers(), secondaries.ref(), input.device_pointers());

    // Launch a kernel to create track initializers from secondaries
    track_init.extend_from_secondaries(&states, &params);

    // Check the vacancies
    output.vacancy   = vacancies_test(track_init.device_pointers());
    expected.vacancy = {2, 6};
    EXPECT_VEC_EQ(expected.vacancy, output.vacancy);

    // Check the track IDs of the track initializers created from secondaries
    // Output is sorted as TrackInitializerStore does not calculate IDs
    // deterministically
    output.initializer_id = initializers_test(track_init.device_pointers());
    std::sort(std::begin(output.initializer_id),
              std::end(output.initializer_id));
    expected.initializer_id = {0, 1, 15, 16, 17};
    EXPECT_VEC_EQ(expected.initializer_id, output.initializer_id);

    // Initialize secondaries on device
    track_init.initialize_tracks(&states, &params);

    // Check the track IDs of the initialized tracks
    // Output is sorted as TrackInitializerStore does not calculate IDs
    // deterministically
    output.track_id = tracks_test(states.device_pointers());
    std::sort(std::begin(output.track_id), std::end(output.track_id));
    expected.track_id = {3, 5, 7, 9, 11, 12, 13, 14, 16, 17};
    EXPECT_VEC_EQ(expected.track_id, output.track_id);
}

TEST_F(TrackInitTest, primaries)
{
    const size_type num_tracks = 512;
    const size_type capacity   = 1024;

    // Create primary particles
    const size_type      num_primaries = 8192;
    std::vector<Primary> primaries     = generate_primaries(num_primaries);

    // Allocate storage on device
    StateStore              states({num_tracks, geo_params, 12345u});
    CollectionStateStore<SecondaryAllocatorData, MemSpace::device> secondaries(
        capacity);
    TrackInitializerStore   track_init(num_tracks, capacity, primaries);

    // Kill all the tracks in each interaction and don't produce secondaries
    std::vector<size_type> alloc(num_tracks, 0);
    std::vector<char>      alive(num_tracks, 0);
    ITTestInput            input(alloc, alive);

    for (auto i = num_primaries; i > 0; i -= capacity)
    {
        EXPECT_EQ(track_init.num_primaries(), i);

        // Create track initializers on device from primary particles
        track_init.extend_from_primaries();

        for (auto j = capacity; j > 0; j -= num_tracks)
        {
            EXPECT_EQ(track_init.size(), j);

            // Initialize tracks on device
            track_init.initialize_tracks(&states, &params);

            // Launch kernel that will kill all trackss
            interact(states.device_pointers(),
                     secondaries.ref(),
                     input.device_pointers());

            // Launch a kernel to create track initializers from secondaries
            track_init.extend_from_secondaries(&states, &params);
        }
    }

    // Check the final track IDs
    ITTestOutput output, expected;
    output.track_id = tracks_test(states.device_pointers());
    expected.track_id.resize(num_tracks);
    std::iota(expected.track_id.begin(), expected.track_id.end(), 0);
    EXPECT_VEC_EQ(expected.track_id, output.track_id);

    EXPECT_EQ(track_init.num_primaries(), 0);
    EXPECT_EQ(track_init.size(), 0);
}

TEST_F(TrackInitTest, secondaries)
{
    const size_type num_tracks = 512;
    const size_type capacity   = 1024;

    // Create primary particles
    const size_type      num_primaries = 128;
    std::vector<Primary> primaries     = generate_primaries(num_primaries);

    // Allocate storage on device
    StateStore              states({num_tracks, geo_params, 12345u});
    CollectionStateStore<SecondaryAllocatorData, MemSpace::device> secondaries(
        capacity);
    TrackInitializerStore   track_init(num_tracks, capacity, primaries);

    // Allocate input device data (number of secondaries to produce for each
    // track and whether the track survives the interaction)
    std::vector<size_type> alloc     = {1, 1, 2, 0, 0, 0, 0, 0};
    std::vector<char>      alive     = {1, 0, 0, 1, 0, 0, 0, 0};
    size_type              base_size = alive.size();
    for (size_type i = 0; i < num_tracks / base_size - 1; ++i)
    {
        alloc.insert(alloc.end(), alloc.begin(), alloc.begin() + base_size);
        alive.insert(alive.end(), alive.begin(), alive.begin() + base_size);
    }
    ITTestInput input(alloc, alive);

    // Create track initializers on device from primary particles
    track_init.extend_from_primaries();
    EXPECT_EQ(track_init.num_primaries(), 0);
    EXPECT_EQ(track_init.size(), num_primaries);

    while (track_init.size())
    {
        // Initialize the primary tracks on device
        track_init.initialize_tracks(&states, &params);

        // Launch kernel to process interactions
        interact(states.device_pointers(),
                 secondaries.ref(),
                 input.device_pointers());

        // Launch a kernel to create track initializers from secondaries
        track_init.extend_from_secondaries(&states, &params);
    }
}

//---------------------------------------------------------------------------//
} // namespace celeritas_test
