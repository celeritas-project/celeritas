//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
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

#include "geometry/GeoTestBase.hh"
#include "TrackInit.test.hh"

namespace celeritas_test
{
using namespace celeritas;
using TrackInitDeviceValue
    = TrackInitStateData<Ownership::value, MemSpace::device>;

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

class TrackInitTest : public GeoTestBase<celeritas::GeoParams>
{
  protected:
    const char* dirname() const override { return "sim"; }
    const char* filebase() const override { return "two-boxes"; }

    void SetUp() override
    {
        // Set up shared geometry data
        params.geometry = this->geometry()->device_ref();

        // Set up shared material data
        materials = std::make_shared<MaterialParams>(
            MaterialParams::Input{{{1, units::AmuMass{1.008}, "H"}},
                                  {{1e-5 * constants::na_avogadro,
                                    100.0,
                                    MatterState::gas,
                                    {{ElementId{0}, 1.0}},
                                    "H2"}}});
        params.materials = materials->device_ref();

        // Set up dummy geo/material coupling data
        geo_mats = std::make_shared<GeoMaterialParams>(GeoMaterialParams::Input{
            this->geometry(),
            materials,
            std::vector<MaterialId>(this->geometry()->num_volumes(),
                                    MaterialId{0}),
            {}});
        params.geo_mats = geo_mats->device_ref();

        // Set up shared particle data
        particles = std::make_shared<ParticleParams>(
            ParticleParams::Input{{"gamma",
                                   pdg::gamma(),
                                   zero_quantity(),
                                   zero_quantity(),
                                   ParticleRecord::stable_decay_constant()}});
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

    //! Create primary particles
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

    //! Create mutable state data
    void build_states(size_type num_tracks, size_type storage_factor)
    {
        CELER_EXPECT(params);
        CELER_EXPECT(track_inits);

        ParamsData<Ownership::const_reference, MemSpace::host> host_params;

        host_params.geometry  = this->geometry()->host_ref();
        host_params.geo_mats  = geo_mats->host_ref();
        host_params.materials = materials->host_ref();
        host_params.particles = particles->host_ref();
        host_params.cutoffs   = cutoffs->host_ref();
        host_params.physics   = physics.host();
        host_params.rng       = rng->host_ref();
        host_params.control.secondary_stack_factor = storage_factor;
        CELER_ASSERT(host_params);

        // Allocate state data
        resize(&device_states, host_params, num_tracks);
        states = device_states;

        resize(&track_init_states, track_inits->host_ref(), num_tracks);
        CELER_ENSURE(states && track_init_states);
    }

    //! Copy results to host
    ITTestOutput
    get_result(StateDeviceRef& states, TrackInitDeviceValue& track_init_states)
    {
        CELER_EXPECT(states);
        CELER_EXPECT(track_init_states);

        ITTestOutput result;

        // Copy track initializer data to host
        TrackInitStateData<Ownership::value, MemSpace::host> data;
        data = track_init_states;

        // Store the IDs of the vacant track slots
        const auto vacancies = data.vacancies.data();
        result.vacancies     = {vacancies.begin(), vacancies.end()};

        // Store the track IDs of the initializers
        for (const auto& init : data.initializers.data())
        {
            result.init_ids.push_back(init.sim.track_id.get());
        }

        // Copy sim states to host
        StateCollection<SimTrackState, Ownership::value, MemSpace::host> sim(
            states.sim.state);

        // Store the track IDs and parent IDs
        for (auto tid : range(ThreadId{sim.size()}))
        {
            result.track_ids.push_back(sim[tid].track_id.get());
            result.parent_ids.push_back(sim[tid].parent_id.unchecked_get());
        }

        return result;
    }

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
    TrackInitDeviceValue                          track_init_states;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(TrackInitTest, run)
{
    const size_type num_primaries  = 12;
    const size_type num_tracks     = 10;
    const size_type storage_factor = 10;
    size_type       capacity       = num_tracks * storage_factor;

    // Construct persistent track initializer data
    track_inits = std::make_shared<TrackInitParams>(
        TrackInitParams::Input{generate_primaries(num_primaries), capacity});

    build_states(num_tracks, storage_factor);

    // Check that all of the track slots were marked as empty
    {
        auto result = get_result(states, track_init_states);
        static const unsigned int expected_vacancies[]
            = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        EXPECT_VEC_EQ(expected_vacancies, result.vacancies);
    }

    // Create track initializers on device from primary particles
    extend_from_primaries(track_inits->host_ref(), &track_init_states);

    // Check the track IDs of the track initializers created from primaries
    {
        auto result = get_result(states, track_init_states);
        static const unsigned int expected_track_ids[]
            = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        EXPECT_VEC_EQ(expected_track_ids, result.init_ids);
    }

    // Initialize the primary tracks on device
    initialize_tracks(params, states, &track_init_states);

    // Check the IDs of the initialized tracks
    {
        auto result = get_result(states, track_init_states);
        static const unsigned int expected_track_ids[]
            = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        EXPECT_VEC_EQ(expected_track_ids, result.track_ids);
    }

    // Allocate input device data (number of secondaries to produce for each
    // track and whether the track survives the interaction)
    std::vector<size_type> alloc = {1, 1, 0, 0, 1, 1, 0, 0, 1, 1};
    std::vector<char>      alive = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
    ITTestInput            input(alloc, alive);

    // Launch kernel to process interactions
    interact(states, input.device_ref());

    // Launch a kernel to create track initializers from secondaries
    extend_from_secondaries(params, states, &track_init_states);

    // Check the vacancies
    {
        auto result = get_result(states, track_init_states);
        static const unsigned int expected_vacancies[] = {2, 6};
        EXPECT_VEC_EQ(expected_vacancies, result.vacancies);
    }

    // Check the track IDs of the track initializers created from secondaries
    // Output is sorted as TrackInitializerStore does not calculate IDs
    // deterministically
    {
        auto result = get_result(states, track_init_states);
        std::sort(std::begin(result.init_ids), std::end(result.init_ids));
        static const unsigned int expected_track_ids[] = {0, 1, 13, 15, 17};
        EXPECT_VEC_EQ(expected_track_ids, result.init_ids);
    }

    // Initialize secondaries on device
    initialize_tracks(params, states, &track_init_states);

    // Check the track IDs of the initialized tracks
    // Output is sorted as TrackInitializerStore does not calculate IDs
    // deterministically
    {
        auto         result = get_result(states, track_init_states);
        unsigned int expected_track_ids[]
            = {12, 3, 15, 5, 14, 7, 17, 9, 16, 11};
        std::sort(std::begin(result.track_ids), std::end(result.track_ids));
        std::sort(std::begin(expected_track_ids), std::end(expected_track_ids));
        EXPECT_VEC_EQ(expected_track_ids, result.track_ids);
    }
}

TEST_F(TrackInitTest, primaries)
{
    const size_type num_primaries  = 8192;
    const size_type num_tracks     = 512;
    const size_type storage_factor = 2;
    size_type       capacity       = num_tracks * storage_factor;

    // Construct persistent track initializer data
    track_inits = std::make_shared<TrackInitParams>(
        TrackInitParams::Input{generate_primaries(num_primaries), capacity});

    build_states(num_tracks, storage_factor);

    // Kill all the tracks in each interaction and don't produce secondaries
    std::vector<size_type> alloc(num_tracks, 0);
    std::vector<char>      alive(num_tracks, 0);
    ITTestInput            input(alloc, alive);

    for (auto i = num_primaries; i > 0; i -= capacity)
    {
        EXPECT_EQ(track_init_states.num_primaries, i);

        // Create track initializers on device from primary particles
        extend_from_primaries(track_inits->host_ref(), &track_init_states);

        for (auto j = capacity; j > 0; j -= num_tracks)
        {
            EXPECT_EQ(track_init_states.initializers.size(), j);

            // Initialize tracks on device
            initialize_tracks(params, states, &track_init_states);

            // Launch kernel that will kill all trackss
            interact(states, input.device_ref());

            // Launch a kernel to create track initializers from secondaries
            extend_from_secondaries(params, states, &track_init_states);
        }
    }

    // Check the final track IDs
    auto                   result = get_result(states, track_init_states);
    std::vector<size_type> expected_track_ids(num_tracks);
    std::iota(expected_track_ids.begin(), expected_track_ids.end(), 0);
    EXPECT_VEC_EQ(expected_track_ids, result.track_ids);

    EXPECT_EQ(track_init_states.num_primaries, 0);
    EXPECT_EQ(track_init_states.initializers.size(), 0);
}

TEST_F(TrackInitTest, secondaries)
{
    const size_type num_primaries  = 128;
    const size_type num_tracks     = 512;
    const size_type storage_factor = 2;
    size_type       capacity       = num_tracks * storage_factor;

    // Construct persistent track initializer data
    track_inits = std::make_shared<TrackInitParams>(
        TrackInitParams::Input{generate_primaries(num_primaries), capacity});

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
    extend_from_primaries(track_inits->host_ref(), &track_init_states);
    EXPECT_EQ(track_init_states.num_primaries, 0);
    EXPECT_EQ(track_init_states.initializers.size(), num_primaries);

    while (track_init_states.initializers.size() > 0)
    {
        // Initialize the primary tracks on device
        initialize_tracks(params, states, &track_init_states);

        // Launch kernel to process interactions
        interact(states, input.device_ref());

        // Launch a kernel to create track initializers from secondaries
        extend_from_secondaries(params, states, &track_init_states);
    }
}

//---------------------------------------------------------------------------//
} // namespace celeritas_test
