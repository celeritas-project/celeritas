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
#include "physics/base/Units.hh"

using namespace celeritas;

namespace demo_interactor
{
//---------------------------------------------------------------------------//
/*!
 * Construct with parameters.
 */
KNDemoRunner::KNDemoRunner(constSPParticleParams particles,
                           constSPXsGridParams   xs,
                           CudaGridParams        solver)
    : pparams_(std::move(particles))
    , xsparams_(std::move(xs))
    , launch_params_(std::move(solver))
{
    CELER_EXPECT(pparams_);
    CELER_EXPECT(xsparams_);
    CELER_EXPECT(launch_params_.block_size > 0);

    // Set up KN interactor data;
    namespace pdg            = celeritas::pdg;
    kn_pointers_.model_id    = ModelId{0}; // Unused but needed for error check
    kn_pointers_.electron_id = pparams_->find(pdg::electron());
    kn_pointers_.gamma_id    = pparams_->find(pdg::gamma());
    kn_pointers_.inv_electron_mass
        = 1 / pparams_->get(kn_pointers_.electron_id).mass().value();
    CELER_ENSURE(kn_pointers_);
}

//---------------------------------------------------------------------------//
/*!
 * Run with a given particle vector size and max iterations.
 */
auto KNDemoRunner::operator()(KNDemoRunArgs args) -> result_type
{
    CELER_EXPECT(args.energy > 0);
    CELER_EXPECT(args.num_tracks > 0);

    // Initialize results
    result_type result;
    result.time.reserve(args.max_steps);
    result.alive.reserve(args.max_steps + 1);
    result.edep.reserve(args.max_steps);

    // Start timer for overall execution
    Stopwatch total_time;

    // Particle data
    // TODO: refactor these as collections, and simplify
    RngStateStore           rng_states(args.num_tracks, args.seed);
    DeviceVector<Real3>     position(args.num_tracks);
    DeviceVector<Real3>     direction(args.num_tracks);
    DeviceVector<double>    time(args.num_tracks);
    DeviceVector<bool>      alive(args.num_tracks);

    ParticleStateData<Ownership::value, MemSpace::device> track_states;
    resize(&track_states, pparams_->host_pointers(), args.num_tracks);

    // Secondary data
    StackAllocatorData<Secondary, Ownership::value, MemSpace::device> secondaries;
    resize(&secondaries, args.num_tracks);

    // Detector data
    DetectorParamsData detector_params;
    detector_params.tally_grid = args.tally_grid;
    DetectorStateData<Ownership::value, MemSpace::device> detector_states;
    resize(&detector_states, detector_params, args.num_tracks);

    // Construct pointers to device data
    ParamsDeviceRef params;
    params.particle      = pparams_->device_pointers();
    params.tables        = xsparams_->device_pointers();
    params.kn_interactor = kn_pointers_;
    params.detector      = detector_params;

    InitialPointers initial;
    initial.particle = ParticleTrackState{kn_pointers_.gamma_id,
                                          units::MevEnergy{args.energy}};

    StateDeviceRef state;
    state.particle  = track_states;
    state.rng       = rng_states.device_pointers();
    state.position  = position.device_pointers();
    state.direction = direction.device_pointers();
    state.time      = time.device_pointers();
    state.alive     = alive.device_pointers();

    state.secondaries = secondaries;
    state.detector    = detector_states;

    // Initialize particle states
    initialize(launch_params_, params, state, initial);
    result.alive.push_back(args.num_tracks);

    size_type remaining_steps = args.max_steps;
    while (result.alive.back())
    {
        // Launch the kernel
        Stopwatch elapsed_time;
        demo_interactor::iterate(launch_params_, params, state);

        // Save the wall time
        if (launch_params_.sync)
        {
            result.time.push_back(elapsed_time());
        }

        // Process detector hits and clear secondaries
        demo_interactor::cleanup(launch_params_, params, state);

        // Calculate and save number of living particles
        result.alive.push_back(demo_interactor::reduce_alive(
            launch_params_, alive.device_pointers()));

        if (--remaining_steps == 0)
        {
            // Exceeded step count
            break;
        }
    }

    // Copy integrated energy deposition
    result.edep.resize(detector_params.tally_grid.size);
    demo_interactor::finalize(params, state, make_span(result.edep));

    // Store total time
    result.total_time = total_time();

    return result;
}

//---------------------------------------------------------------------------//
} // namespace demo_interactor
