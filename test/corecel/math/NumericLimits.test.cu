//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! file NumericLimits.test.cu
//---------------------------------------------------------------------------//
#include "NumericLimits.test.hh"

#include "corecel/DeviceRuntimeApi.hh"

#include "corecel/Assert.hh"
#include "corecel/math/NumericLimits.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/KernelParamCalculator.device.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//

template<class T>
__global__ void nl_test_kernel(NLTestOutput<T>* data)
{
    using limits_t = numeric_limits<T>;
    unsigned int local_thread_id = KernelParamCalculator::thread_id().get();
    if (local_thread_id == 0)
    {
        data->eps = limits_t::epsilon();
    }
    else if (local_thread_id == 1)
    {
        data->nan = limits_t::quiet_NaN();
    }
    else if (local_thread_id == 2)
    {
        data->inf = limits_t::infinity();
    }
    else if (local_thread_id == 3)
    {
        data->max = limits_t::max();
    }
    else if (local_thread_id == 4)
    {
        data->inv_zero = T(1) / T(0);
    }
}

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Run on device and return results
template<class T>
NLTestOutput<T> nl_test()
{
    // Allocate output data
    NLTestOutput<T>* result_device;
    CELER_DEVICE_API_CALL(Malloc(&result_device, sizeof(NLTestOutput<T>)));

    static KernelParamCalculator const calc_launch_params(
        "nl_test", nl_test_kernel<T>, device().threads_per_warp());
    auto grid = calc_launch_params(4);

    CELER_LAUNCH_KERNEL_IMPL(nl_test_kernel<T>,
                             grid.blocks_per_grid,
                             grid.threads_per_block,
                             0,
                             0,
                             result_device);
    CELER_DEVICE_API_CALL(DeviceSynchronize());

    // Copy to host
    NLTestOutput<T> result;
    CELER_DEVICE_API_CALL(Memcpy(&result,
                                 result_device,
                                 sizeof(NLTestOutput<T>),
                                 CELER_DEVICE_API_SYMBOL(MemcpyDeviceToHost)));
    CELER_DEVICE_API_CALL(Free(result_device));
    return result;
}

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//

template NLTestOutput<float> nl_test<float>();
template NLTestOutput<double> nl_test<double>();

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
