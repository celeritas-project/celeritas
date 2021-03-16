//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file HostKNDemoRunner.cc
//---------------------------------------------------------------------------//
#include "HostKNDemoRunner.hh"

#include <iostream>
#include <random>
#include "base/ArrayUtils.hh"
#include "base/CollectionStateStore.hh"
#include "base/Range.hh"
#include "base/StackAllocator.hh"
#include "base/Stopwatch.hh"
#include "random/distributions/ExponentialDistribution.hh"
#include "physics/base/ParticleTrackView.hh"
#include "physics/base/Units.hh"
#include "physics/base/Secondary.hh"
#include "physics/em/detail/KleinNishinaInteractor.hh"
#include "physics/grid/XsCalculator.hh"
#include "KernelUtils.hh"
#include "Detector.hh"

using namespace celeritas;
using celeritas::detail::KleinNishinaInteractor;

namespace demo_interactor
{
//---------------------------------------------------------------------------//
/*!
 * Construct with parameters.
 */
HostKNDemoRunner::HostKNDemoRunner(constSPParticleParams particles,
                                   constSPXsGridParams   xs)
    : pparams_(std::move(particles)), xsparams_(std::move(xs))
{
    CELER_EXPECT(pparams_);
    CELER_EXPECT(xsparams_);

    // Set up KN interactor data;
    kn_pointers_.model_id    = ModelId{0}; // Unused but needed for error check
    kn_pointers_.electron_id = pparams_->find(pdg::electron());
    kn_pointers_.gamma_id    = pparams_->find(pdg::gamma());
    kn_pointers_.inv_electron_mass
        = 1 / pparams_->get(kn_pointers_.electron_id).mass().value();
    CELER_ENSURE(kn_pointers_);
}

//---------------------------------------------------------------------------//
/*!
 * Run given number of particles each for max steps.
 */
auto HostKNDemoRunner::operator()(demo_interactor::KNDemoRunArgs args)
    -> result_type
{
    CELER_EXPECT(args.energy > 0);
    CELER_EXPECT(args.num_tracks > 0);

    // Initialize results
    result_type result;
    result.time.reserve(args.max_steps);
    result.alive.resize(args.max_steps + 1);
    result.edep.reserve(args.max_steps);

    // Start timer for overall execution and transport-only time
    Stopwatch total_time;
    double    transport_time = 0.0;

    // Random number generations
    std::mt19937 rng(args.seed);

    // Particle data
    ParticleStateData<Ownership::value, MemSpace::host> track_states;
    resize(&track_states, pparams_->host_pointers(), 1);

    // Make secondary store
    StackAllocatorData<Secondary, Ownership::value, MemSpace::host> secondaries;
    resize(&secondaries, args.max_steps);

    // Detector data
    DetectorParamsData detector_params;
    detector_params.tally_grid = args.tally_grid;
    DetectorStateData<Ownership::value, MemSpace::host> detector_states;
    resize(&detector_states, detector_params, args.max_steps);

    // Construct references
    ParamsHostRef params;
    params.particle      = pparams_->host_pointers();
    params.tables        = xsparams_->host_pointers();
    params.kn_interactor = kn_pointers_;
    params.detector      = detector_params;

    // Construct initialization
    InitialPointers initial;
    initial.particle = ParticleTrackState{kn_pointers_.gamma_id,
                                          units::MevEnergy{args.energy}};

    StateHostRef state;
    state.particle    = track_states;
    state.secondaries = secondaries;
    state.detector    = detector_states;

    // Loop over particle tracks and events per track
    for (CELER_MAYBE_UNUSED auto n : range(args.num_tracks))
    {
        // Storage for track state
        Real3     position  = {0, 0, 0};
        Real3     direction = {0, 0, 1};
        real_type time      = 0;
        bool      alive     = true;

        // Create and initialize particle view
        ParticleTrackView particle(
            params.particle, state.particle, ThreadId{0});

        // Create helper classes
        StackAllocator<Secondary> allocate_secondaries(state.secondaries);
        Detector                  detector(params.detector, state.detector);
        XsCalculator calc_xs(params.tables.xs, params.tables.reals);

        CELER_ASSERT(state.secondaries.capacity() == args.max_steps);
        CELER_ASSERT(state.detector.hit_buffer.capacity() == args.max_steps);
        CELER_ASSERT(allocate_secondaries.get().size() == 0);
        CELER_ASSERT(detector.num_hits() == 0);

        // Counters
        size_type num_steps = 0;
        auto      remaining_steps = args.max_steps;
        Stopwatch elapsed_time;

        particle = initial.particle;

        while (alive && --remaining_steps > 0)
        {
            // Increment alive counter
            CELER_ASSERT(num_steps < result.alive.size());
            result.alive[num_steps]++;
            ++num_steps;

            // Move to collision
            demo_interactor::move_to_collision(
                particle, calc_xs, direction, &position, &time, rng);

            // Hit analysis
            Hit h;
            h.pos    = position;
            h.dir    = direction;
            h.thread = ThreadId(0);
            h.time   = time;

            // Check for below energy cutoff
            if (particle.energy() < units::MevEnergy{0.01})
            {
                // Particle is below interaction energy
                h.energy_deposited = particle.energy();

                // Deposit energy and kill
                detector.buffer_hit(h);
                alive = false;
                continue;
            }

            // Construct the KN interactor
            KleinNishinaInteractor interact(
                kn_pointers_, particle, direction, allocate_secondaries);

            // Perform interactions - emits a single particle
            Interaction interaction = interact(rng);
            CELER_ASSERT(interaction);
            CELER_ASSERT(interaction.secondaries.size() == 1);

            // Deposit energy from the secondary (all local)
            {
                const auto& secondary = interaction.secondaries.front();
                h.dir                 = secondary.direction;
                h.energy_deposited    = secondary.energy;
                detector.buffer_hit(h);
            }

            // Update the energy and direction in the state from the
            // interaction
            direction = interaction.direction;
            particle.energy(interaction.energy);
        }
        CELER_ASSERT(num_steps < args.max_steps
                         ? allocate_secondaries.get().size() == num_steps - 1
                         : allocate_secondaries.get().size() == num_steps);
        CELER_ASSERT(detector.num_hits() == num_steps);

        // Store transport time
        transport_time += elapsed_time();

        // Clear secondaries
        allocate_secondaries.clear();
        CELER_ASSERT(allocate_secondaries.get().size() == 0);

        // Bin the tally results from the buffer onto the grid
        for (auto hit_id : range(Detector::HitId{detector.num_hits()}))
        {
            detector.process_hit(hit_id);
        }
        detector.clear_buffer();
    }

    // Copy integrated energy deposition
    result.edep.resize(detector_params.tally_grid.size);
    demo_interactor::finalize(params, state, make_span(result.edep));

    // Store timings
    result.time.push_back(transport_time);
    result.total_time = total_time();

    // Remove trailing zeros from preallocated "alive" size
    while (!result.alive.empty() && result.alive.back() == 0)
    {
        result.alive.pop_back();
    }

    return result;
}

//---------------------------------------------------------------------------//
} // namespace demo_interactor
