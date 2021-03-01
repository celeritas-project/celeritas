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
#include "base/PieStateStore.hh"
#include "base/Range.hh"
#include "base/Stopwatch.hh"
#include "random/distributions/ExponentialDistribution.hh"
#include "physics/base/ParticleTrackView.hh"
#include "physics/base/Units.hh"
#include "physics/base/Secondary.hh"
#include "physics/base/SecondaryAllocatorView.hh"
#include "physics/em/detail/KleinNishinaInteractor.hh"
#include "physics/grid/PhysicsGridCalculator.hh"
#include "DetectorView.hh"
#include "HostStackAllocatorStore.hh"
#include "HostDetectorStore.hh"
#include "KernelUtils.hh"

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

    // Physics calculator
    const auto&           xs_host_ptrs = xsparams_->host_pointers();
    PhysicsGridCalculator calc_xs(xs_host_ptrs.xs, xs_host_ptrs.reals);

    // Make secondary store
    HostStackAllocatorStore<Secondary> secondaries(args.max_steps);
    auto secondary_host_ptrs = secondaries.host_pointers();

    // Make detector store
    HostDetectorStore detector(args.max_steps, args.tally_grid);
    auto              detector_host_ptrs = detector.host_pointers();

    // Particle state store
    PieStateStore<ParticleStateData, MemSpace::host> particle_state(*pparams_,
                                                                    1);

    // Loop over particle tracks and events per track
    for (CELER_MAYBE_UNUSED auto n : range(args.num_tracks))
    {
        // Place cap on maximum number of steps
        auto remaining_steps = args.max_steps;

        // Make the initial track state
        ParticleTrackState init_state
            = {kn_pointers_.gamma_id, units::MevEnergy(args.energy)};

        // Initialize particle state
        Real3     position  = {0, 0, 0};
        Real3     direction = {0, 0, 1};
        real_type time      = 0;
        bool      alive     = true;

        // Create and initialize particle view
        ParticleTrackView particle(
            pparams_->host_pointers(), particle_state.ref(), ThreadId{0});
        particle = init_state;

        // Secondary pointers
        SecondaryAllocatorView allocate_secondaries(secondary_host_ptrs);
        CELER_ASSERT(secondaries.capacity() == args.max_steps);
        CELER_ASSERT(allocate_secondaries.get().size() == 0);

        // Detector hits
        DetectorView detector_hit(detector_host_ptrs);

        // Step counter
        size_type num_steps = 0;

        Stopwatch elapsed_time;
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
            h.thread = ThreadId(0);
            h.time   = time;

            // Check for below energy cutoff
            if (particle.energy() < units::MevEnergy{0.01})
            {
                // Particle is below interaction energy
                h.dir              = direction;
                h.energy_deposited = particle.energy();

                // Deposit energy and kill
                detector_hit(h);
                alive = false;
                continue;
            }

            // Construct the KN interactor
            KleinNishinaInteractor interact(
                kn_pointers_, particle, direction, allocate_secondaries);

            // Perform interactions - emits a single particle
            auto interaction = interact(rng);
            CELER_ASSERT(interaction);
            CELER_ASSERT(interaction.secondaries.size() == 1);

            // Deposit energy from the secondary (all local)
            {
                const auto& secondary = interaction.secondaries.front();
                h.dir                 = secondary.direction;
                h.energy_deposited    = secondary.energy;
                detector_hit(h);
            }

            // Update the energy and direction in the state from the
            // interaction
            direction = interaction.direction;
            particle.energy(interaction.energy);
        }
        CELER_ASSERT(num_steps < args.max_steps
                         ? secondaries.get_size() == num_steps - 1
                         : secondaries.get_size() == num_steps);
        CELER_ASSERT(secondaries.get_size()
                     == allocate_secondaries.get().size());
        CELER_ASSERT(
            StackAllocatorView<Hit>(detector.host_pointers().hit_buffer)
                .get()
                .size()
            == num_steps);

        // Store transport time
        transport_time += elapsed_time();

        // Clear secondaries
        secondaries.clear();
        CELER_ASSERT(secondaries.get_size() == 0);

        // Bin the tally results from the buffer onto the grid
        detector.bin_buffer();
    }

    // Copy integrated energy deposition
    result.edep = detector.finalize(1 / real_type(args.num_tracks));

    // Store timings
    result.time.push_back(transport_time);
    result.total_time = total_time();

    // Reduce "alive" size
    while (!result.alive.empty() && result.alive.back() == 0)
    {
        result.alive.pop_back();
    }

    return result;
}

//---------------------------------------------------------------------------//
} // namespace demo_interactor
