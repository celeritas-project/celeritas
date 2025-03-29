//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/random/RngEngine.test.cu
//---------------------------------------------------------------------------//

#include "celeritas/random/RngEngine.hh"

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include "corecel/DeviceRuntimeApi.hh"

#include "corecel/sys/Device.hh"
#include "corecel/sys/KernelParamCalculator.device.hh"

#include "RngEngine.test.hh"

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

__global__ void sample_native_kernel(RngDeviceParamsRef params,
                                     RngDeviceStateRef states,
                                     RngEngine::result_type* samples)
{
    auto tid = TrackSlotId{KernelParamCalculator::thread_id().unchecked_get()};
    if (tid.get() < states.size())
    {
        RngEngine rng(params, states, tid);
        samples[tid.get()] = rng();
    }
}

template<class RealType>
__global__ void sample_canonical_kernel(RngDeviceParamsRef params,
                                        RngDeviceStateRef states,
                                        RealType* samples)
{
    auto tid = TrackSlotId{KernelParamCalculator::thread_id().unchecked_get()};
    if (tid.get() < states.size())
    {
        RngEngine rng(params, states, tid);
        samples[tid.get()] = generate_canonical<RealType>(rng);
    }
}
}  // namespace

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Run on device and return results
std::vector<unsigned int>
re_test_native(RngDeviceParamsRef params, RngDeviceStateRef states)
{
    thrust::device_vector<unsigned int> samples(states.size());

    CELER_LAUNCH_KERNEL(sample_native,
                        states.size(),
                        0,
                        params,
                        states,
                        raw_pointer_cast(samples.data()));

    std::vector<unsigned int> host_samples(states.size());
    thrust::copy(samples.begin(), samples.end(), host_samples.begin());

    return host_samples;
}

//---------------------------------------------------------------------------//
//! Run on device and return results
template<class T>
std::vector<T>
re_test_canonical(RngDeviceParamsRef params, RngDeviceStateRef states)
{
    thrust::device_vector<T> samples(states.size());

    static KernelParamCalculator const calc_launch_params(
        "sample_canonical", sample_canonical_kernel<T>);
    auto grid = calc_launch_params(states.size());

    CELER_LAUNCH_KERNEL_IMPL(sample_canonical_kernel<T>,
                             grid.blocks_per_grid,
                             grid.threads_per_block,
                             0,
                             0,
                             params,
                             states,
                             raw_pointer_cast(samples.data()));
    CELER_DEVICE_API_CALL(DeviceSynchronize());

    std::vector<T> host_samples(states.size());
    thrust::copy(samples.begin(), samples.end(), host_samples.begin());

    return host_samples;
}

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//

template std::vector<float>
    re_test_canonical<float>(RngDeviceParamsRef, RngDeviceStateRef);
template std::vector<double>
    re_test_canonical<double>(RngDeviceParamsRef, RngDeviceStateRef);

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
