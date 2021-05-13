//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file TrackInit.test.cc
//---------------------------------------------------------------------------//
#include "sim/TrackInitUtils.hh"

#include <algorithm>
#include <numeric>
#include "celeritas_test.hh"
#include "base/CollectionStateStore.hh"
#include "geometry/GeoParams.hh"
#include "physics/base/ParticleParams.hh"
#include "physics/material/MaterialParams.hh"
#include "random/RngParams.hh"
#include "sim/TrackInitParams.hh"
#include "sim/TrackInterface.hh"
#include "TrackInit.test.hh"

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
        material_params = std::make_shared<MaterialParams>(std::move(mats));

        particle_params = std::make_shared<ParticleParams>(
            ParticleParams::Input{{"gamma",
                                   pdg::gamma(),
                                   zero_quantity(),
                                   zero_quantity(),
                                   ParticleDef::stable_decay_constant()}});

        rng_params = std::make_shared<RngParams>(12345);
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

    // Create shared problem data
    void build_params(size_type num_primaries, size_type storage_factor)
    {
        // Construct persistent track initializer data
        TrackInitParams::Input inp{generate_primaries(num_primaries),
                                   storage_factor};
        init_params = std::make_shared<TrackInitParams>(std::move(inp));

        params.geometry  = geo_params->device_pointers();
        params.materials = material_params->device_pointers();
        params.particles = particle_params->device_pointers();
        params.rng       = rng_params->device_pointers();
        CELER_ENSURE(params);
    }

    // Create mutable state data
    void build_states(size_type num_tracks, size_type storage_factor)
    {
        CELER_EXPECT(params);
        CELER_EXPECT(init_params);

        // Allocate storage for secondaries on device
        secondaries
            = CollectionStateStore<SecondaryAllocatorData, MemSpace::device>(
                num_tracks * storage_factor);

        ParamsData<Ownership::const_reference, MemSpace::host> host_params;
        host_params.geometry  = geo_params->host_pointers();
        host_params.materials = material_params->host_pointers();
        host_params.particles = particle_params->host_pointers();
        host_params.rng       = rng_params->host_pointers();
        CELER_ASSERT(host_params);

        // Allocate state data
        resize(&init, init_params->host_pointers(), num_tracks);
        resize(&device_states, host_params, num_tracks);
        states = device_states;
        CELER_ENSURE(states);
    }

    std::shared_ptr<GeoParams>       geo_params;
    std::shared_ptr<ParticleParams>  particle_params;
    std::shared_ptr<MaterialParams>  material_params;
    std::shared_ptr<RngParams>       rng_params;
    std::shared_ptr<TrackInitParams> init_params;

    CollectionStateStore<SecondaryAllocatorData, MemSpace::device> secondaries;
    StateData<Ownership::value, MemSpace::device> device_states;

    ParamsDeviceRef         params;
    StateDeviceRef          states;
    TrackInitStateDeviceVal init;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(TrackInitTest, run)
{
    const size_type num_primaries  = 12;
    const size_type num_tracks     = 10;
    const size_type storage_factor = 10;

    build_params(num_primaries, storage_factor);
    build_states(num_tracks, storage_factor);

    // Check that all of the track slots were marked as empty
    ITTestOutput output, expected;
    output.vacancy   = vacancies_test(make_ref(init));
    expected.vacancy = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    EXPECT_VEC_EQ(expected.vacancy, output.vacancy);

    // Create track initializers on device from primary particles
    extend_from_primaries(init_params->host_pointers(), &init);

    // Check the track IDs of the track initializers created from primaries
    output.initializer_id   = initializers_test(make_ref(init));
    expected.initializer_id = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    EXPECT_VEC_EQ(expected.initializer_id, output.initializer_id);

    // Initialize the primary tracks on device
    initialize_tracks(params, states, &init);

    // Check the IDs of the initialized tracks
    output.track_id   = tracks_test(states);
    expected.track_id = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    EXPECT_VEC_EQ(expected.track_id, output.track_id);

    // Allocate input device data (number of secondaries to produce for each
    // track and whether the track survives the interaction)
    std::vector<size_type> alloc = {1, 1, 0, 0, 1, 1, 0, 0, 1, 1};
    std::vector<char>      alive = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
    ITTestInput            input(alloc, alive);

    // Launch kernel to process interactions
    interact(states, secondaries.ref(), input.device_pointers());

    // Launch a kernel to create track initializers from secondaries
    extend_from_secondaries(params, states, &init);

    // Check the vacancies
    output.vacancy   = vacancies_test(make_ref(init));
    expected.vacancy = {2, 6};
    EXPECT_VEC_EQ(expected.vacancy, output.vacancy);

    // Check the track IDs of the track initializers created from secondaries
    // Output is sorted as TrackInitializerStore does not calculate IDs
    // deterministically
    output.initializer_id = initializers_test(make_ref(init));
    std::sort(std::begin(output.initializer_id),
              std::end(output.initializer_id));
    expected.initializer_id = {0, 1, 15, 16, 17};
    EXPECT_VEC_EQ(expected.initializer_id, output.initializer_id);

    // Initialize secondaries on device
    initialize_tracks(params, states, &init);

    // Check the track IDs of the initialized tracks
    // Output is sorted as TrackInitializerStore does not calculate IDs
    // deterministically
    output.track_id   = tracks_test(states);
    expected.track_id = {12, 3, 16, 5, 13, 7, 17, 9, 14, 11};
    std::sort(std::begin(output.track_id), std::end(output.track_id));
    std::sort(std::begin(expected.track_id), std::end(expected.track_id));
    EXPECT_VEC_EQ(expected.track_id, output.track_id);
}

TEST_F(TrackInitTest, primaries)
{
    const size_type num_primaries  = 8192;
    const size_type num_tracks     = 512;
    const size_type storage_factor = 2;
    size_type       capacity       = num_tracks * storage_factor;

    build_params(num_primaries, storage_factor);
    build_states(num_tracks, storage_factor);

    // Kill all the tracks in each interaction and don't produce secondaries
    std::vector<size_type> alloc(num_tracks, 0);
    std::vector<char>      alive(num_tracks, 0);
    ITTestInput            input(alloc, alive);

    for (auto i = num_primaries; i > 0; i -= capacity)
    {
        EXPECT_EQ(init.num_primaries, i);

        // Create track initializers on device from primary particles
        extend_from_primaries(init_params->host_pointers(), &init);

        for (auto j = capacity; j > 0; j -= num_tracks)
        {
            EXPECT_EQ(init.initializers.size(), j);

            // Initialize tracks on device
            initialize_tracks(params, states, &init);

            // Launch kernel that will kill all trackss
            interact(states, secondaries.ref(), input.device_pointers());

            // Launch a kernel to create track initializers from secondaries
            extend_from_secondaries(params, states, &init);
        }
    }

    // Check the final track IDs
    ITTestOutput output, expected;
    output.track_id = tracks_test(states);
    expected.track_id.resize(num_tracks);
    std::iota(expected.track_id.begin(), expected.track_id.end(), 0);
    EXPECT_VEC_EQ(expected.track_id, output.track_id);

    EXPECT_EQ(init.num_primaries, 0);
    EXPECT_EQ(init.initializers.size(), 0);
}

TEST_F(TrackInitTest, secondaries)
{
    const size_type num_primaries  = 128;
    const size_type num_tracks     = 512;
    const size_type storage_factor = 2;

    build_params(num_primaries, storage_factor);
    build_states(num_tracks, storage_factor);

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
    extend_from_primaries(init_params->host_pointers(), &init);
    EXPECT_EQ(init.num_primaries, 0);
    EXPECT_EQ(init.initializers.size(), num_primaries);

    while (init.initializers.size() > 0)
    {
        // Initialize the primary tracks on device
        initialize_tracks(params, states, &init);

        // Launch kernel to process interactions
        interact(states, secondaries.ref(), input.device_pointers());

        // Launch a kernel to create track initializers from secondaries
        extend_from_secondaries(params, states, &init);
    }
}

//---------------------------------------------------------------------------//
} // namespace celeritas_test
