//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file physics/base/Particle.test.cu
//---------------------------------------------------------------------------//
#include "Particle.test.hh"

#include <thrust/device_vector.h>

#include "corecel/device_runtime_api.h"
#include "corecel/sys/KernelParamCalculator.device.hh"
#include "corecel/sys/Device.hh"
#include "celeritas/phys/ParticleTrackView.hh"

using thrust::raw_pointer_cast;

namespace celeritas_test
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//

__global__ void ptv_test_kernel(unsigned int                    size,
                                ParticleParamsRef               params,
                                ParticleStateRef                states,
                                const ParticleTrackInitializer* init,
                                double*                         result)
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
    thrust::device_vector<ParticleTrackInitializer> init = input.init;

    thrust::device_vector<double> result(init.size()
                                         * PTVTestOutput::props_per_thread());

    CELER_LAUNCH_KERNEL(ptv_test,
                        celeritas::device().default_block_size(),
                        init.size(),
                        init.size(),
                        input.params,
                        input.states,
                        raw_pointer_cast(init.data()),
                        raw_pointer_cast(result.data()));
    CELER_DEVICE_CALL_PREFIX(DeviceSynchronize());

    PTVTestOutput output;
    output.props.resize(result.size());
    thrust::copy(result.begin(), result.end(), output.props.begin());
    return output;
}

//---------------------------------------------------------------------------//
} // namespace celeritas_test
