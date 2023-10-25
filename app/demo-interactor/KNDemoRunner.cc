//---------------------------------*-C++-*-----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-interactor/KNDemoRunner.cc
//---------------------------------------------------------------------------//
#include "KNDemoRunner.hh"

#include "corecel/cont/Range.hh"
#include "corecel/sys/Stopwatch.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/random/RngParams.hh"

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Construct with parameters.
 */
KNDemoRunner::KNDemoRunner(constSPParticleParams particles,
                           constSPXsGridParams xs,
                           DeviceGridParams solver)
    : pparams_(std::move(particles))
    , xsparams_(std::move(xs))
    , launch_params_(std::move(solver))
{
    CELER_EXPECT(pparams_);
    CELER_EXPECT(xsparams_);

    // Set up KN interactor data;
    namespace pdg = pdg;
    kn_data_.ids.action = ActionId{0};  // Unused but needed for error check
    kn_data_.ids.electron = pparams_->find(pdg::electron());
    kn_data_.ids.gamma = pparams_->find(pdg::gamma());
    kn_data_.inv_electron_mass
        = 1 / pparams_->get(kn_data_.ids.electron).mass().value();
    CELER_ENSURE(kn_data_);
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
    DeviceVector<Real3> position(args.num_tracks);
    DeviceVector<Real3> direction(args.num_tracks);
    DeviceVector<real_type> time(args.num_tracks);
    DeviceVector<bool> alive(args.num_tracks);

    ParticleStateData<Ownership::value, MemSpace::device> track_states;
    resize(&track_states, pparams_->host_ref(), args.num_tracks);

    RngParams rng_params(args.seed);
    RngStateData<Ownership::value, MemSpace::device> rng_states;
    resize(&rng_states, rng_params.host_ref(), StreamId{0}, args.num_tracks);

    // Secondary data
    StackAllocatorData<Secondary, Ownership::value, MemSpace::device> secondaries;
    resize(&secondaries, args.num_tracks);

    // Detector data
    DetectorParamsData detector_params;
    detector_params.tally_grid = args.tally_grid;
    DetectorStateData<Ownership::value, MemSpace::device> detector_states;
    resize(&detector_states, detector_params, args.num_tracks);

    // Construct data to device data
    DeviceCRef<ParamsData> params;
    params.particle = pparams_->device_ref();
    params.tables = xsparams_->device_ref();
    params.kn_interactor = kn_data_;
    params.detector = detector_params;

    InitialData initial;
    initial.particle = ParticleTrackInitializer{kn_data_.ids.gamma,
                                                units::MevEnergy(args.energy)};

    DeviceRef<StateData> state;
    state.particle = track_states;
    state.rng = rng_states;
    state.position = position.device_ref();
    state.direction = direction.device_ref();
    state.time = time.device_ref();
    state.alive = alive.device_ref();

    state.secondaries = secondaries;
    state.detector = detector_states;

    // Initialize particle states
    initialize(launch_params_, params, state, initial);
    result.alive.push_back(args.num_tracks);

    size_type remaining_steps = args.max_steps;
    while (result.alive.back())
    {
        // Launch the kernel
        Stopwatch elapsed_time;
        celeritas::app::iterate(launch_params_, params, state);

        // Save the wall time
        if (launch_params_.sync)
        {
            result.time.push_back(elapsed_time());
        }

        // Process detector hits and clear secondaries
        celeritas::app::cleanup(launch_params_, params, state);

        // Calculate and save number of living particles
        result.alive.push_back(
            celeritas::app::reduce_alive(launch_params_, alive.device_ref()));

        if (--remaining_steps == 0)
        {
            // Exceeded step count
            break;
        }
    }

    // Copy integrated energy deposition
    std::vector<real_type> edep(detector_params.tally_grid.size);
    celeritas::app::finalize(params, state, make_span(edep));
    result.edep.assign(edep.begin(), edep.end());

    // Store total time
    result.total_time = total_time();

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
