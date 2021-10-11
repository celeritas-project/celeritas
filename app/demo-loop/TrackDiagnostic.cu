//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file TrackDiagnostic.cu
//---------------------------------------------------------------------------//
#include "TrackDiagnostic.hh"
#include "base/Assert.hh"
#include "base/KernelParamCalculator.cuda.hh"
#include "sim/SimTrackView.hh"
#include "physics/base/ModelInterface.hh"

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

using namespace celeritas;

namespace demo_loop
{
//---------------------------------------------------------------------------//
// KERNELS
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
size_type
reduce_alive(const StateData<Ownership::reference, MemSpace::device>& states)
{
    auto sim_states = states.sim.state[AllItems<SimTrackState>{}].data();

    return thrust::transform_reduce(
        thrust::device_pointer_cast(sim_states),
        thrust::device_pointer_cast(sim_states) + states.size(),
        alive(),
        0,
        thrust::plus<size_type>());
}

//---------------------------------------------------------------------------//
} // namespace demo_loop
