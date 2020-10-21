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
#include "DetectorStore.hh"

using namespace celeritas;

namespace demo_interactor
{
//---------------------------------------------------------------------------//
/*!
 * Construct with parameters.
 */
KNDemoRunner::KNDemoRunner(constSPParticleParams     particles,
                           constSPPhysicsArrayParams xs,
                           CudaGridParams            solver)
    : pparams_(std::move(particles))
    , xsparams_(std::move(xs))
    , launch_params_(std::move(solver))
{
    REQUIRE(pparams_);
    REQUIRE(xsparams_);
    REQUIRE(launch_params_.block_size > 0);
    REQUIRE(launch_params_.grid_size > 0);

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
    DeviceVector<Real3>     position(args.num_tracks);
    DeviceVector<Real3>     direction(args.num_tracks);
    DeviceVector<double>    time(args.num_tracks);
    DeviceVector<bool>      alive(args.num_tracks);
    DetectorStore           detector(args.num_tracks, args.tally_grid);

    // Construct pointers to device data
    ParamPointers params;
    params.particle      = pparams_->device_pointers();
    params.xs            = xsparams_->device_pointers();
    params.kn_interactor = kn_pointers_;

    InitialPointers initial;
    initial.particle = ParticleTrackState{kn_pointers_.gamma_id,
                                          units::MevEnergy{args.energy}};

    StatePointers state;
    state.particle  = track_states.device_pointers();
    state.rng       = rng_states.device_pointers();
    state.position  = position.device_pointers();
    state.direction = direction.device_pointers();
    state.time      = time.device_pointers();
    state.alive     = alive.device_pointers();

    // Initialize particle states
    initialize(launch_params_, params, state, initial);
    result.alive.push_back(args.num_tracks);

    size_type remaining_steps = args.max_steps;
    while (result.alive.back())
    {
        // Launch the kernel
        Stopwatch elapsed_time;
        iterate(launch_params_,
                params,
                state,
                secondaries.device_pointers(),
                detector.device_pointers());

        // Save the wall time
        result.time.push_back(elapsed_time());

        // Clear secondaries, which have all effectively been "killed" inside
        // the `iterate` kernel (local energy deposited)
        secondaries.clear();

        // Bin detector depositions from the buffer into the grid
        detector.bin_buffer();

        // Calculate and save number of living particles
        result.alive.push_back(reduce_alive(alive.device_pointers()));

        if (--remaining_steps == 0)
        {
            // Exceeded step count
            break;
        }
    }

    // Copy integrated energy deposition
    result.edep = detector.finalize(1 / real_type(args.num_tracks));

    // Store total time
    result.total_time = total_time();

    return result;
}

//---------------------------------------------------------------------------//
} // namespace demo_interactor
