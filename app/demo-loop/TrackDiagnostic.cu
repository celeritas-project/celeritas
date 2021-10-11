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
 * Sample mean free path and calculate physics step limits.
 */

// __global__
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
