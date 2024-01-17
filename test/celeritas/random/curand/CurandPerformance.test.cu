//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/random/curand/CurandPerformance.test.cu
//---------------------------------------------------------------------------//
#include "CurandPerformance.test.hh"

#include <curand_kernel.h>
#include <curand_mtgp32.h>
#include <curand_mtgp32_host.h>
#include <curand_mtgp32dc_p_11213.h>
#include <thrust/device_vector.h>

#include "corecel/Assert.hh"

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
template<class T>
__global__ void curand_setup_kernel(T* devStates, unsigned long seed)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, tid, 0, &devStates[tid]);
}

template<class T>
__global__ void curand_test_kernel(unsigned int nsamples,
                                   T* devStates,
                                   double* sum,
                                   double* sum2)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int sid = tid;

    T localState = devStates[sid];

    while (tid < nsamples)
    {
        double u01 = curand_uniform(&localState);
        sum[sid] += u01;
        sum2[sid] += u01 * u01;
        tid += blockDim.x * gridDim.x;
    }

    devStates[sid] = localState;
}

__global__ void curand_test_mtgp32_kernel(unsigned int nsamples,
                                          curandStateMtgp32* devStates,
                                          double* sum,
                                          double* sum2)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int sid = tid;

    while (tid < nsamples)
    {
        double u01 = curand_uniform(&devStates[blockIdx.x]);
        sum[sid] += u01;
        sum2[sid] += u01 * u01;
        tid += blockDim.x * gridDim.x;
    }
}
//---------------------------------------------------------------------------//
}  // namespace

//! Run on device and return results
template<class T>
TestOutput curand_test(TestParams params)
{
    // Initialize curand states
    T* devStates = nullptr;
    cudaMalloc(&devStates, params.nthreads * params.nblocks * sizeof(T));

    curand_setup_kernel<<<params.nblocks, params.nthreads>>>(devStates,
                                                             time(NULL));
    CELER_CUDA_CALL(cudaDeviceSynchronize());

    // Output data for kernel
    thrust::device_vector<double> sum(params.nthreads * params.nblocks, 0.0);
    thrust::device_vector<double> sum2(params.nthreads * params.nblocks, 0.0);

    curand_test_kernel<<<params.nblocks, params.nthreads>>>(
        params.nsamples,
        devStates,
        raw_pointer_cast(sum.data()),
        raw_pointer_cast(sum2.data()));
    CELER_CUDA_CALL(cudaDeviceSynchronize());

    // Clean up
    CELER_CUDA_CALL(cudaFree(devStates));

    // Copy result back to CPU
    TestOutput result;

    result.sum.resize(sum.size());
    thrust::copy(sum.begin(), sum.end(), result.sum.begin());

    result.sum2.resize(sum2.size());
    thrust::copy(sum2.begin(), sum2.end(), result.sum2.begin());

    return result;
}

//! Mersenne Twister RNG
template<>
TestOutput curand_test<curandStateMtgp32>(TestParams params)
{
    // Initialize curand states
    curandStateMtgp32* devStates = nullptr;
    mtgp32_kernel_params* kp = nullptr;

    CELER_CUDA_CALL(cudaMalloc((void**)&devStates,
                               params.nblocks * sizeof(curandStateMtgp32)));

    CELER_CUDA_CALL(cudaMalloc((void**)&kp, sizeof(mtgp32_kernel_params)));
    curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, kp);
    curandMakeMTGP32KernelState(
        devStates, mtgp32dc_params_fast_11213, kp, params.nblocks, time(NULL));

    // Output data for kernel
    thrust::device_vector<double> sum(params.nthreads * params.nblocks, 0.0);
    thrust::device_vector<double> sum2(params.nthreads * params.nblocks, 0.0);

    curand_test_mtgp32_kernel<<<params.nblocks, params.nthreads>>>(
        params.nsamples,
        devStates,
        raw_pointer_cast(sum.data()),
        raw_pointer_cast(sum2.data()));
    CELER_CUDA_CALL(cudaDeviceSynchronize());

    // Clean up
    CELER_CUDA_CALL(cudaFree(devStates));

    // Copy result back to CPU
    TestOutput result;

    result.sum.resize(sum.size());
    thrust::copy(sum.begin(), sum.end(), result.sum.begin());

    result.sum2.resize(sum2.size());
    thrust::copy(sum2.begin(), sum2.end(), result.sum2.begin());

    return result;
}

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//
template TestOutput curand_test<curandState>(TestParams);
template TestOutput curand_test<curandStateMRG32k3a>(TestParams);
template TestOutput curand_test<curandStatePhilox4_32_10_t>(TestParams);
template TestOutput curand_test<curandStateMtgp32>(TestParams);

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
