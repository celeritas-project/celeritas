//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/random/RngEngine.test.cu
//---------------------------------------------------------------------------//

#include "celeritas/random/RngEngine.hh"

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include "corecel/device_runtime_api.h"
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

__global__ void sample_native_kernel(DeviceRef<RngStateData> view,
                                     RngEngine::result_type* samples)
{
    auto tid = TrackSlotId{KernelParamCalculator::thread_id().unchecked_get()};
    if (tid.get() < view.size())
    {
        RngEngine rng(view, tid);
        samples[tid.get()] = rng();
    }
}

template<class RealType>
__global__ void
sample_canonical_kernel(DeviceRef<RngStateData> view, RealType* samples)
{
    auto tid = TrackSlotId{KernelParamCalculator::thread_id().unchecked_get()};
    if (tid.get() < view.size())
    {
        RngEngine rng(view, tid);
        samples[tid.get()] = generate_canonical<RealType>(rng);
    }
}
}  // namespace

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Run on device and return results
std::vector<unsigned int> re_test_native(RngDeviceRef states)
{
    thrust::device_vector<unsigned int> samples(states.size());

    CELER_LAUNCH_KERNEL(sample_native,
                        states.size(),
                        0,
                        states,
                        raw_pointer_cast(samples.data()));

    std::vector<unsigned int> host_samples(states.size());
    thrust::copy(samples.begin(), samples.end(), host_samples.begin());

    return host_samples;
}

//---------------------------------------------------------------------------//
//! Run on device and return results
template<class T>
std::vector<T> re_test_canonical(RngDeviceRef states)
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
                             states,
                             raw_pointer_cast(samples.data()));
    CELER_DEVICE_CHECK_ERROR();
    CELER_DEVICE_CALL_PREFIX(DeviceSynchronize());

    std::vector<T> host_samples(states.size());
    thrust::copy(samples.begin(), samples.end(), host_samples.begin());

    return host_samples;
}

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//

template std::vector<float> re_test_canonical<float>(RngDeviceRef);
template std::vector<double> re_test_canonical<double>(RngDeviceRef);

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
