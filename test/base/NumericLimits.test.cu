//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file NumericLimits.test.cu
//---------------------------------------------------------------------------//
#include "NumericLimits.test.hh"

#include "base/Assert.hh"
#include "base/KernelParamCalculator.cuda.hh"
#include "base/NumericLimits.hh"

namespace celeritas_test
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//

template<class T>
__global__ void nl_test_kernel(NLTestOutput<T>* data)
{
    using limits_t = celeritas::numeric_limits<T>;
    unsigned int local_thread_id
        = celeritas::KernelParamCalculator::thread_id().get();
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
    CELER_CUDA_CALL(cudaMalloc(&result_device, sizeof(NLTestOutput<T>)));

    celeritas::KernelParamCalculator calc_launch_params;

    auto params = calc_launch_params(3);
    nl_test_kernel<<<params.grid_size, params.block_size>>>(result_device);
    CELER_CUDA_CALL(cudaDeviceSynchronize());

    // Copy to host
    NLTestOutput<T> result;
    CELER_CUDA_CALL(cudaMemcpy(&result,
                               result_device,
                               sizeof(NLTestOutput<T>),
                               cudaMemcpyDeviceToHost));
    CELER_CUDA_CALL(cudaFree(result_device));
    return result;
}

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//

template NLTestOutput<float>  nl_test<float>();
template NLTestOutput<double> nl_test<double>();

//---------------------------------------------------------------------------//
} // namespace celeritas_test
