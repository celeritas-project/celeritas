//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file TrackDiagnostic.cu
//---------------------------------------------------------------------------//
#include "TrackDiagnostic.hh"

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

#include "base/Assert.hh"
#include "base/KernelParamCalculator.device.hh"
#include "physics/base/ModelData.hh"
#include "sim/SimTrackView.hh"

using namespace celeritas;

namespace demo_loop
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Sums the number of 'alive' tracks.
 *
 * Host-side function using thrust and a functor (third argument) summing alive
 * tracks on the device.
 *
 * Note that a simple thrust::reduce(), without specifying the execution
 * policy, defaults to host memory and therefore causes a memory access
 * segfault; specifying the thrust::device policy leads to compile-time errors
 * due to incompatible arguments.
 */
size_type reduce_alive(const StateDeviceRef& states)
{
    auto sim_states = states.sim.state[AllItems<SimTrackState>{}].data();

    return thrust::transform_reduce(
        thrust::device_pointer_cast(sim_states),
        thrust::device_pointer_cast(sim_states) + states.size(),
        OneIfAlive(),
        0,
        thrust::plus<size_type>());
}
//---------------------------------------------------------------------------//
} // namespace

//---------------------------------------------------------------------------//
/*!
 * Add the current step's number of alive tracks to this diagnostic.
 */
template<>
void TrackDiagnostic<MemSpace::device>::end_step(const StateRef& states)
{
    // Get the number of tracks in flight.
    num_alive_per_step_.push_back(reduce_alive(states));
}

//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//
} // namespace demo_loop
