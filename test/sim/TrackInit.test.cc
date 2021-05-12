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
#include "sim/TrackInterface.hh"
#include "sim/StateStore.hh"
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

    // Create track initializer data
    void build_track_init(size_type num_primaries,
                          size_type storage_factor,
                          size_type num_tracks)
    {
        // Construct persistent data
        TrackInitParams::Input inp{generate_primaries(num_primaries),
                                   storage_factor};
        init_params = std::make_shared<TrackInitParams>(std::move(inp));

        // Allocate state data
        resize(&init_states, init_params->host_pointers(), num_tracks);

        // Allocate storage on device
        states = StateStore({num_tracks, geo_params, 12345u});
        secondaries
            = CollectionStateStore<SecondaryAllocatorData, MemSpace::device>(
                num_tracks * storage_factor);
    }

    ParamStore                                                     params;
    StateStore                                                     states;
    std::shared_ptr<GeoParams>                                     geo_params;
    std::shared_ptr<TrackInitParams>                               init_params;
    TrackInitStateDeviceVal                                        init_states;
    CollectionStateStore<SecondaryAllocatorData, MemSpace::device> secondaries;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(TrackInitTest, run)
{
    const size_type num_primaries  = 12;
    const size_type num_tracks     = 10;
    const size_type storage_factor = 10;

    build_track_init(num_primaries, storage_factor, num_tracks);

    // Check that all of the track slots were marked as empty
    ITTestOutput output, expected;
    output.vacancy   = vacancies_test(make_ref(init_states));
    expected.vacancy = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    EXPECT_VEC_EQ(expected.vacancy, output.vacancy);

    // Create track initializers on device from primary particles
    extend_from_primaries(init_params->host_pointers(), &init_states);

    // Check the track IDs of the track initializers created from primaries
    output.initializer_id   = initializers_test(make_ref(init_states));
    expected.initializer_id = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    EXPECT_VEC_EQ(expected.initializer_id, output.initializer_id);

    // Initialize the primary tracks on device
    initialize_tracks(params, &states, &init_states);

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
    extend_from_secondaries(params, &states, &init_states);

    // Check the vacancies
    output.vacancy   = vacancies_test(make_ref(init_states));
    expected.vacancy = {2, 6};
    EXPECT_VEC_EQ(expected.vacancy, output.vacancy);

    // Check the track IDs of the track initializers created from secondaries
    // Output is sorted as TrackInitializerStore does not calculate IDs
    // deterministically
    output.initializer_id = initializers_test(make_ref(init_states));
    std::sort(std::begin(output.initializer_id),
              std::end(output.initializer_id));
    expected.initializer_id = {0, 1, 15, 16, 17};
    EXPECT_VEC_EQ(expected.initializer_id, output.initializer_id);

    // Initialize secondaries on device
    initialize_tracks(params, &states, &init_states);

    // Check the track IDs of the initialized tracks
    // Output is sorted as TrackInitializerStore does not calculate IDs
    // deterministically
    output.track_id   = tracks_test(states.device_pointers());
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

    build_track_init(num_primaries, storage_factor, num_tracks);

    // Kill all the tracks in each interaction and don't produce secondaries
    std::vector<size_type> alloc(num_tracks, 0);
    std::vector<char>      alive(num_tracks, 0);
    ITTestInput            input(alloc, alive);

    for (auto i = num_primaries; i > 0; i -= capacity)
    {
        EXPECT_EQ(init_states.num_primaries, i);

        // Create track initializers on device from primary particles
        extend_from_primaries(init_params->host_pointers(), &init_states);

        for (auto j = capacity; j > 0; j -= num_tracks)
        {
            EXPECT_EQ(init_states.initializers.size(), j);

            // Initialize tracks on device
            initialize_tracks(params, &states, &init_states);

            // Launch kernel that will kill all trackss
            interact(states.device_pointers(),
                     secondaries.ref(),
                     input.device_pointers());

            // Launch a kernel to create track initializers from secondaries
            extend_from_secondaries(params, &states, &init_states);
        }
    }

    // Check the final track IDs
    ITTestOutput output, expected;
    output.track_id = tracks_test(states.device_pointers());
    expected.track_id.resize(num_tracks);
    std::iota(expected.track_id.begin(), expected.track_id.end(), 0);
    EXPECT_VEC_EQ(expected.track_id, output.track_id);

    EXPECT_EQ(init_states.num_primaries, 0);
    EXPECT_EQ(init_states.initializers.size(), 0);
}

TEST_F(TrackInitTest, secondaries)
{
    const size_type num_primaries  = 128;
    const size_type num_tracks     = 512;
    const size_type storage_factor = 2;

    build_track_init(num_primaries, storage_factor, num_tracks);

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
    extend_from_primaries(init_params->host_pointers(), &init_states);
    EXPECT_EQ(init_states.num_primaries, 0);
    EXPECT_EQ(init_states.initializers.size(), num_primaries);

    while (init_states.initializers.size())
    {
        // Initialize the primary tracks on device
        initialize_tracks(params, &states, &init_states);

        // Launch kernel to process interactions
        interact(states.device_pointers(),
                 secondaries.ref(),
                 input.device_pointers());

        // Launch a kernel to create track initializers from secondaries
        extend_from_secondaries(params, &states, &init_states);
    }
}

//---------------------------------------------------------------------------//
} // namespace celeritas_test
