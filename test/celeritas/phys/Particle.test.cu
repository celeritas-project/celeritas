//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/Particle.test.cu
//---------------------------------------------------------------------------//
#include "Particle.test.hh"

#include <thrust/device_vector.h>

#include "corecel/DeviceRuntimeApi.hh"

#include "corecel/math/Quantity.hh"
#include "corecel/math/UnitUtils.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/KernelParamCalculator.device.hh"
#include "celeritas/UnitTypes.hh"
#include "celeritas/phys/ParticleTrackView.hh"

using thrust::raw_pointer_cast;

namespace celeritas
{
namespace test
{
namespace
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//

__global__ void ptv_test_kernel(unsigned int size,
                                DeviceCRef<ParticleParamsData> params,
                                DeviceRef<ParticleStateData> states,
                                ParticleTrackInitializer const* init,
                                double* result)
{
    using InvSecDecay = RealQuantity<UnitInverse<units::Second>>;

    auto local_tid
        = TrackSlotId{KernelParamCalculator::thread_id().unchecked_get()};
    if (!(local_tid < size))
        return;

    // Initialize particle
    ParticleTrackView p(params, states, local_tid);
    p = init[local_tid.get()];

    // Skip result to the start for this thread
    result += local_tid.get() * PTVTestOutput::props_per_thread();

    // Calculate/write values from the track view
    CELER_ASSERT(p.particle_id() == init[local_tid.get()].particle_id);
    *result++ = p.energy().value();
    *result++ = p.mass().value();
    *result++ = p.charge().value();
    *result++ = native_value_to<InvSecDecay>(p.decay_constant()).value();
    *result++ = p.speed().value();
    *result++ = (p.mass() > zero_quantity() ? p.lorentz_factor() : -1);
    *result++ = p.momentum().value();
    *result++ = p.momentum_sq().value();
}
//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
//! Run on device and return results
PTVTestOutput ptv_test(PTVTestInput input)
{
    thrust::device_vector<ParticleTrackInitializer> init = input.init;

    thrust::device_vector<double> result(init.size()
                                         * PTVTestOutput::props_per_thread());

    CELER_LAUNCH_KERNEL(ptv_test,
                        init.size(),
                        0,
                        init.size(),
                        input.params,
                        input.states,
                        raw_pointer_cast(init.data()),
                        raw_pointer_cast(result.data()));
    CELER_DEVICE_API_CALL(DeviceSynchronize());

    PTVTestOutput output;
    output.props.resize(result.size());
    thrust::copy(result.begin(), result.end(), output.props.begin());
    return output;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
