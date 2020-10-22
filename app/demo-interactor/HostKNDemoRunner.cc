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
#include "base/Stopwatch.hh"
#include "base/Range.hh"
#include "base/ArrayUtils.hh"
#include "random/distributions/ExponentialDistribution.hh"
#include "physics/base/ParticleTrackView.hh"
#include "physics/base/Units.hh"
#include "physics/base/Secondary.hh"
#include "physics/base/SecondaryAllocatorView.hh"
#include "physics/em/KleinNishinaInteractor.hh"
#include "PhysicsArrayCalculator.hh"
#include "DetectorView.hh"
#include "HostStackAllocatorStore.hh"
#include "HostDetectorStore.hh"

using namespace celeritas;

namespace demo_interactor
{
//---------------------------------------------------------------------------//
/*!
 * \brief Construct with parameters.
 */
HostKNDemoRunner::HostKNDemoRunner(constSPParticleParams     particles,
                                   constSPPhysicsArrayParams xs)
    : pparams_(std::move(particles)), xsparams_(std::move(xs))
{
    REQUIRE(pparams_);
    REQUIRE(xsparams_);

    // Set up KN interactor data;
    namespace pdg            = celeritas::pdg;
    kn_pointers_.electron_id = pparams_->find(pdg::electron());
    kn_pointers_.gamma_id    = pparams_->find(pdg::gamma());
    kn_pointers_.inv_electron_mass
        = 1 / pparams_->get(kn_pointers_.electron_id).mass.value();
    ENSURE(kn_pointers_);
}

//---------------------------------------------------------------------------//
/*!
 * \brief Run given number of particles each for max steps.
 */
auto HostKNDemoRunner::operator()(demo_interactor::KNDemoRunArgs args)
    -> result_type
{
    REQUIRE(args.energy > 0);
    REQUIRE(args.num_tracks > 0);

    // Initialize results
    result_type result;
    result.time.reserve(args.max_steps);
    result.alive.reserve(args.max_steps + 1);
    result.edep.reserve(args.max_steps);

    // Start timer for overall execution and transport-only time
    Stopwatch total_time;
    double    transport_time = 0.0;

    // Random number generations
    std::mt19937 rng(args.seed);

    // Particle param pointers
    auto pp_host_ptrs = pparams_->host_pointers();

    // Physics calculator
    auto                   xs_host_ptrs = xsparams_->host_pointers();
    PhysicsArrayCalculator calc_xs(xs_host_ptrs);

    // Make secondary store
    HostStackAllocatorStore<Secondary> secondaries(args.max_steps);
    auto secondary_host_ptrs = secondaries.host_pointers();

    // Make detector store
    HostDetectorStore detector(args.max_steps, args.tally_grid);
    auto              detector_host_ptrs = detector.host_pointers();

    // Loop over particle tracks and events per track
    for (CELER_MAYBE_UNUSED auto n : celeritas::range(args.num_tracks))
    {
        // Place cap on maximum number of steps
        auto remaining_steps = args.max_steps;

        // Make the initial track state
        celeritas::ParticleTrackState init_state = {
            kn_pointers_.gamma_id, celeritas::units::MevEnergy(args.energy)};

        // Initialize particle state
        StatePointers state;
        state.particle.vars = {&init_state, 1};
        state.position      = {0, 0, 0};
        state.direction     = {0, 0, 1};
        state.time          = 0;
        state.alive         = true;

        // Secondary pointers
        SecondaryAllocatorView allocate_secondaries(secondary_host_ptrs);
        CHECK(secondaries.capacity() == args.max_steps);
        CHECK(allocate_secondaries.get().size() == 0);

        // Detector hits
        DetectorView detector_hit(detector_host_ptrs);

        // Step counter
        CELER_MAYBE_UNUSED size_type num_steps = 0;

        Stopwatch elapsed_time;
        while (state.alive && --remaining_steps > 0)
        {
            // Get a particle track view to a single particle
            auto particle
                = ParticleTrackView(pp_host_ptrs, state.particle, ThreadId(0));

            // Move to collision
            {
                real_type                          sigma = calc_xs(particle);
                ExponentialDistribution<real_type> sample_distance(sigma);
                real_type distance = sample_distance(rng);
                celeritas::axpy(distance, state.direction, &state.position);
                state.time += distance * celeritas::unit_cast(particle.speed());
            }

            // Update step counter
            ++num_steps;

            // Hit analysis
            Hit h;
            h.pos    = state.position;
            h.thread = ThreadId(0);
            h.time   = state.time;

            // Check for below energy cutoff
            if (particle.energy()
                < KleinNishinaInteractor::min_incident_energy())
            {
                // Particle is below interaction energy
                h.dir              = state.direction;
                h.energy_deposited = particle.energy();

                // Deposit energy and kill
                detector_hit(h);
                state.alive = false;
                continue;
            }

            // Construct the KN interactor
            KleinNishinaInteractor interact(
                kn_pointers_, particle, state.direction, allocate_secondaries);

            // Perform interactions - emits a single particle
            auto interaction = interact(rng);
            CHECK(interaction);
            CHECK(interaction.secondaries.size() == 1);

            // Deposit energy from the secondary (all local)
            {
                const auto& secondary = interaction.secondaries.front();
                h.dir                 = secondary.direction;
                h.energy_deposited    = secondary.energy;
                detector_hit(h);
            }

            // Update the energy and direction in the state from the
            // interaction
            state.direction = interaction.direction;
            particle.energy(interaction.energy);
        }
        CHECK(num_steps <= args.max_steps);
        CHECK(num_steps < args.max_steps
                  ? secondaries.get_size() == num_steps - 1
                  : secondaries.get_size() == num_steps);
        CHECK(secondaries.get_size() == allocate_secondaries.get().size());
        CHECK(StackAllocatorView<Hit>(detector.host_pointers().hit_buffer)
                  .get()
                  .size()
              == num_steps);

        // Store transport time
        transport_time += elapsed_time();

        // Clear secondaries
        secondaries.clear();
        CHECK(secondaries.get_size() == 0);

        // Bin the tally results from the buffer onto the grid
        detector.bin_buffer();
    }

    // Copy integrated energy deposition
    result.edep = detector.finalize(1 / real_type(args.num_tracks));

    // Store timings
    result.time.push_back(transport_time);
    result.total_time = total_time();

    return result;
}

//---------------------------------------------------------------------------//
} // namespace demo_interactor