//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RngEngine.test.cu
//---------------------------------------------------------------------------//

#include "random/RngEngine.hh"

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include "base/device_runtime_api.h"
#include "base/KernelParamCalculator.device.hh"
#include "comm/Device.hh"

#include "RngEngine.test.hh"

using namespace celeritas;
using thrust::raw_pointer_cast;

namespace celeritas_test
{
namespace
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//

__global__ void
sample_native_kernel(RngStateData<Ownership::reference, MemSpace::device> view,
                     RngEngine::result_type* samples)
{
    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (tid.get() < view.size())
    {
        RngEngine rng(view, tid);
        samples[tid.get()] = rng();
    }
}

template<class RealType>
__global__ void
sample_canonical_kernel(RngStateData<Ownership::reference, MemSpace::device> view,
                        RealType* samples)
{
    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (tid.get() < view.size())
    {
        RngEngine rng(view, tid);
        samples[tid.get()] = generate_canonical<RealType>(rng);
    }
}
} // namespace

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Run on device and return results
std::vector<unsigned int> re_test_native(RngDeviceRef states)
{
    thrust::device_vector<unsigned int> samples(states.size());

    CELER_LAUNCH_KERNEL(sample_native,
                        celeritas::device().default_block_size(),
                        states.size(),
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

    static const ::celeritas::KernelParamCalculator calc_launch_params(
        sample_canonical_kernel<T>,
        "sample_canonical",
        celeritas::device().default_block_size());
    auto grid = calc_launch_params(states.size());

    CELER_LAUNCH_KERNEL_IMPL(sample_canonical_kernel<T>,
                             grid.grid_size,
                             grid.block_size,
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

template std::vector<float>  re_test_canonical<float>(RngDeviceRef);
template std::vector<double> re_test_canonical<double>(RngDeviceRef);

//---------------------------------------------------------------------------//
} // namespace celeritas_test
