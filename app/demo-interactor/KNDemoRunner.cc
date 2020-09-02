//---------------------------------*-C++-*-----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file KNDemoRunner.cc
//---------------------------------------------------------------------------//
#include "KNDemoRunner.hh"

#include "base/Range.hh"
#include "base/Stopwatch.hh"
#include "random/cuda/RngStateStore.hh"
#include "physics/base/ParticleStateStore.hh"
#include "physics/base/SecondaryAllocatorStore.hh"
#include "physics/base/Units.hh"

using namespace celeritas;

namespace demo_interactor
{
//---------------------------------------------------------------------------//
/*!
 * Construct with parameters.
 */
KNDemoRunner::KNDemoRunner(constSPParticleParams particles,
                           CudaGridParams        solver)
    : pparams_(std::move(particles)), launch_params_(std::move(solver))
{
    REQUIRE(pparams_);
    REQUIRE(launch_params_.block_size > 0);
    REQUIRE(launch_params_.grid_size > 0);

    // Set up KN interactor data;
    using celeritas::units::speed_of_light_sq;
    namespace pdg            = celeritas::pdg;
    kn_pointers_.electron_id = pparams_->find(pdg::electron());
    kn_pointers_.gamma_id    = pparams_->find(pdg::gamma());
    kn_pointers_.inv_electron_mass_csq
        = 1
          / (pparams_->get(kn_pointers_.electron_id).mass * speed_of_light_sq);
    ENSURE(kn_pointers_);
}

//---------------------------------------------------------------------------//
/*!
 * Run with a given particle vector size and max iterations.
 */
auto KNDemoRunner::operator()(KNDemoRunArgs args) -> result_type
{
    REQUIRE(args.energy > 0);
    REQUIRE(args.num_tracks > 0);

    // Initialize results
    result_type result;
    result.time.reserve(args.max_steps);
    result.alive.reserve(args.max_steps + 1);
    result.edep.reserve(args.max_steps);

    // Start timer for overall execution
    Stopwatch total_time;

    // Allocate device data
    SecondaryAllocatorStore secondaries(args.num_tracks);
    ParticleStateStore      track_states(args.num_tracks);
    RngStateStore           rng_states(args.num_tracks, args.seed);
    DeviceVector<Real3>     direction(args.num_tracks);
    DeviceVector<double>    energy_deposition(args.num_tracks);
    DeviceVector<bool>      alive(args.num_tracks);

    // Initialize particle states
    initialize(launch_params_,
               pparams_->device_pointers(),
               track_states.device_pointers(),
               ParticleTrackState{kn_pointers_.gamma_id, args.energy},
               rng_states.device_pointers(),
               direction.device_pointers(),
               alive.device_pointers());

    result.alive.push_back(args.num_tracks);

    size_type remaining_steps = args.max_steps;
    while (result.alive.back())
    {
        // Launch the kernel
        Stopwatch elapsed_time;
        iterate(launch_params_,
                pparams_->device_pointers(),
                track_states.device_pointers(),
                kn_pointers_,
                secondaries.device_pointers(),
                rng_states.device_pointers(),
                direction.device_pointers(),
                alive.device_pointers(),
                energy_deposition.device_pointers());

        // Save the time
        result.time.push_back(elapsed_time());

        // Calculate average energy deposition
        result.edep.push_back(
            reduce_energy_dep(energy_deposition.device_pointers()));

        // Clear secondaries, which have all effectively been "killed" inside
        // the `iterate` kernel (local energy deposited)
        secondaries.clear();

        // Calculate and save number of living particles
        result.alive.push_back(reduce_alive(alive.device_pointers()));

        if (--remaining_steps == 0)
        {
            // Exceeded step count
            break;
        }
    }

    // Store total time
    result.total_time = total_time();

    return result;
}

//---------------------------------------------------------------------------//
} // namespace demo_interactor
