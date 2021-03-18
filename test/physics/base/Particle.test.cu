//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Particle.test.cu
//---------------------------------------------------------------------------//
#include "physics/base/ParticleTrackView.hh"
#include "Particle.test.hh"

#include <thrust/device_vector.h>
#include "base/KernelParamCalculator.cuda.hh"

using thrust::raw_pointer_cast;

namespace celeritas_test
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//

__global__ void ptv_test_kernel(unsigned int              size,
                                ParticleParamsPointers    params,
                                ParticleStatePointers     states,
                                const ParticleTrackState* init,
                                double*                   result)
{
    auto local_thread_id = celeritas::KernelParamCalculator::thread_id();
    if (!(local_thread_id < size))
        return;

    // Initialize particle
    ParticleTrackView p(params, states, local_thread_id);
    p = init[local_thread_id.get()];

    // Skip result to the start for this thread
    result += local_thread_id.get() * PTVTestOutput::props_per_thread();

    // Calculate/write values from the track view
    CELER_ASSERT(p.particle_id() == init[local_thread_id.get()].particle_id);
    *result++ = p.energy().value();
    *result++ = p.mass().value();
    *result++ = p.charge().value();
    *result++ = p.decay_constant();
    *result++ = p.speed().value();
    *result++ = (p.mass() > zero_quantity() ? p.lorentz_factor() : -1);
    *result++ = p.momentum().value();
    *result++ = p.momentum_sq().value();
}

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Run on device and return results
PTVTestOutput ptv_test(PTVTestInput input)
{
    thrust::device_vector<ParticleTrackState> init = input.init;
    thrust::device_vector<double>             result(init.size()
                                         * PTVTestOutput::props_per_thread());

    static const celeritas::KernelParamCalculator calc_launch_params(
        ptv_test_kernel, "ptv_test");
    auto params = calc_launch_params(init.size());
    ptv_test_kernel<<<params.grid_size, params.block_size>>>(
        init.size(),
        input.params,
        input.states,
        raw_pointer_cast(init.data()),
        raw_pointer_cast(result.data()));
    CELER_CUDA_CHECK_ERROR();
    CELER_CUDA_CALL(cudaDeviceSynchronize());

    PTVTestOutput output;
    output.props.resize(result.size());
    thrust::copy(result.begin(), result.end(), output.props.begin());
    return output;
}

//---------------------------------------------------------------------------//
} // namespace celeritas_test
