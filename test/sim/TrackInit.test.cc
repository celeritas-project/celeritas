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
#include "geometry/GeoMaterialParams.hh"
#include "physics/base/CutoffParams.hh"
#include "physics/base/ParticleParams.hh"
#include "physics/base/PhysicsParams.hh"
#include "physics/material/MaterialParams.hh"
#include "random/RngParams.hh"
#include "sim/TrackInitParams.hh"
#include "sim/TrackData.hh"
#include "TrackInit.test.hh"

namespace celeritas_test
{
using namespace celeritas;

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

ITTestInputData ITTestInput::device_ref()
{
    ITTestInputData result;
    result.alloc_size = alloc_size.device_ref();
    result.alive      = alive.device_ref();
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
        geometry            = std::make_shared<GeoParams>(test_file.c_str());
        params.geometry     = geometry->device_ref();

        // Set up shared material data
        materials = std::make_shared<MaterialParams>(
            MaterialParams::Input{{{1, units::AmuMass{1.008}, "H"}},
                                  {{1e-5 * constants::na_avogadro,
                                    100.0,
                                    MatterState::gas,
                                    {{ElementId{0}, 1.0}},
                                    "H2"}}});
        params.materials = materials->device_ref();

        // Set up dummy geometry/material coupling data
        geo_mats = std::make_shared<GeoMaterialParams>(GeoMaterialParams::Input{
            geometry,
            materials,
            std::vector<MaterialId>(geometry->num_volumes(), MaterialId{0})});
        params.geo_mats = geo_mats->device_ref();

        // Set up shared particle data
        particles = std::make_shared<ParticleParams>(
            ParticleParams::Input{{"gamma",
                                   pdg::gamma(),
                                   zero_quantity(),
                                   zero_quantity(),
                                   ParticleDef::stable_decay_constant()}});
        params.particles = particles->device_ref();

        // Set up empty cutoff data
        cutoffs = std::make_shared<CutoffParams>(
            CutoffParams::Input{particles, materials, {}});
        params.cutoffs = cutoffs->device_ref();

        // Set up shared RNG data
        rng            = std::make_shared<RngParams>(12345);
        params.rng     = rng->device_ref();

        // Add dummy physics data
        PhysicsParamsData<Ownership::value, MemSpace::host> host_physics;
        resize(&host_physics.process_groups, 1);
        host_physics.max_particle_processes = 1;
        host_physics.scaling_min_range      = 1;
        host_physics.scaling_fraction       = 0.2;
        host_physics.energy_fraction        = 0.8;
        host_physics.linear_loss_limit      = 0.01;
        physics = CollectionMirror<PhysicsParamsData>{std::move(host_physics)};
        params.physics = physics.device();

        CELER_ENSURE(params);
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

    // Create mutable state data
    void build_states(size_type num_tracks, size_type storage_factor)
    {
        CELER_EXPECT(params);
        CELER_EXPECT(track_inits);

        ParamsData<Ownership::const_reference, MemSpace::host> host_params;
        host_params.geometry                       = geometry->host_ref();
        host_params.geo_mats                       = geo_mats->host_ref();
        host_params.materials                      = materials->host_ref();
        host_params.particles                      = particles->host_ref();
        host_params.cutoffs                        = cutoffs->host_ref();
        host_params.physics     = physics.host();
        host_params.rng                            = rng->host_ref();
        host_params.track_inits                    = track_inits->host_ref();
        host_params.control.secondary_stack_factor = storage_factor;
        CELER_ASSERT(host_params);

        // Allocate state data
        resize(&device_states.track_inits, track_inits->host_ref(), num_tracks);
        resize(&device_states, host_params, num_tracks);
        states = device_states;
        CELER_ENSURE(states);
    }

    std::shared_ptr<GeoParams>                    geometry;
    std::shared_ptr<GeoMaterialParams>            geo_mats;
    std::shared_ptr<ParticleParams>               particles;
    std::shared_ptr<MaterialParams>               materials;
    std::shared_ptr<CutoffParams>                 cutoffs;
    std::shared_ptr<RngParams>                    rng;
    std::shared_ptr<TrackInitParams>              track_inits;
    CollectionMirror<PhysicsParamsData>           physics;
    StateData<Ownership::value, MemSpace::device> device_states;
    ParamsDeviceRef                               params;
    StateDeviceRef                                states;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(TrackInitTest, run)
{
    const size_type num_primaries  = 12;
    const size_type num_tracks     = 10;
    const size_type storage_factor = 10;

    // Construct persistent track initializer data
    track_inits = std::make_shared<TrackInitParams>(TrackInitParams::Input{
        generate_primaries(num_primaries), storage_factor});

    build_states(num_tracks, storage_factor);

    // Check that all of the track slots were marked as empty
    ITTestOutput output, expected;
    output.vacancy   = vacancies_test(make_ref(device_states.track_inits));
    expected.vacancy = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    EXPECT_VEC_EQ(expected.vacancy, output.vacancy);

    // Create track initializers on device from primary particles
    extend_from_primaries(track_inits->host_ref(), &device_states.track_inits);

    // Check the track IDs of the track initializers created from primaries
    output.init_id   = initializers_test(make_ref(device_states.track_inits));
    expected.init_id = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    EXPECT_VEC_EQ(expected.init_id, output.init_id);

    // Initialize the primary tracks on device
    initialize_tracks(params, states, &device_states.track_inits);

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
    interact(states, input.device_ref());

    // Launch a kernel to create track initializers from secondaries
    extend_from_secondaries(params, states, &device_states.track_inits);

    // Check the vacancies
    output.vacancy   = vacancies_test(make_ref(device_states.track_inits));
    expected.vacancy = {2, 6};
    EXPECT_VEC_EQ(expected.vacancy, output.vacancy);

    // Check the track IDs of the track initializers created from secondaries
    // Output is sorted as TrackInitializerStore does not calculate IDs
    // deterministically
    output.init_id = initializers_test(make_ref(device_states.track_inits));
    std::sort(std::begin(output.init_id), std::end(output.init_id));
    expected.init_id = {0, 1, 15, 16, 17};
    EXPECT_VEC_EQ(expected.init_id, output.init_id);

    // Initialize secondaries on device
    initialize_tracks(params, states, &device_states.track_inits);

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

    // Construct persistent track initializer data
    track_inits = std::make_shared<TrackInitParams>(TrackInitParams::Input{
        generate_primaries(num_primaries), storage_factor});

    build_states(num_tracks, storage_factor);

    // Kill all the tracks in each interaction and don't produce secondaries
    std::vector<size_type> alloc(num_tracks, 0);
    std::vector<char>      alive(num_tracks, 0);
    ITTestInput            input(alloc, alive);

    for (auto i = num_primaries; i > 0; i -= capacity)
    {
        EXPECT_EQ(device_states.track_inits.num_primaries, i);

        // Create track initializers on device from primary particles
        extend_from_primaries(track_inits->host_ref(),
                              &device_states.track_inits);

        for (auto j = capacity; j > 0; j -= num_tracks)
        {
            EXPECT_EQ(device_states.track_inits.initializers.size(), j);

            // Initialize tracks on device
            initialize_tracks(params, states, &device_states.track_inits);

            // Launch kernel that will kill all trackss
            interact(states, input.device_ref());

            // Launch a kernel to create track initializers from secondaries
            extend_from_secondaries(params, states, &device_states.track_inits);
        }
    }

    // Check the final track IDs
    ITTestOutput output, expected;
    output.track_id = tracks_test(states);
    expected.track_id.resize(num_tracks);
    std::iota(expected.track_id.begin(), expected.track_id.end(), 0);
    EXPECT_VEC_EQ(expected.track_id, output.track_id);

    EXPECT_EQ(device_states.track_inits.num_primaries, 0);
    EXPECT_EQ(device_states.track_inits.initializers.size(), 0);
}

TEST_F(TrackInitTest, secondaries)
{
    const size_type num_primaries  = 128;
    const size_type num_tracks     = 512;
    const size_type storage_factor = 2;

    // Construct persistent track initializer data
    track_inits = std::make_shared<TrackInitParams>(TrackInitParams::Input{
        generate_primaries(num_primaries), storage_factor});

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
    extend_from_primaries(track_inits->host_ref(), &device_states.track_inits);
    EXPECT_EQ(device_states.track_inits.num_primaries, 0);
    EXPECT_EQ(device_states.track_inits.initializers.size(), num_primaries);

    while (device_states.track_inits.initializers.size() > 0)
    {
        // Initialize the primary tracks on device
        initialize_tracks(params, states, &device_states.track_inits);

        // Launch kernel to process interactions
        interact(states, input.device_ref());

        // Launch a kernel to create track initializers from secondaries
        extend_from_secondaries(params, states, &device_states.track_inits);
    }
}

//---------------------------------------------------------------------------//
} // namespace celeritas_test
