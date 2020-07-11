//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file StackAllocator.test.cu
//---------------------------------------------------------------------------//
#include "StackAllocator.test.hh"

#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include "base/KernelParamCalculator.cuda.hh"
#include "base/StackAllocator.hh"

using thrust::raw_pointer_cast;

namespace celeritas_test
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//
// Allocate based on input values, incrementing thread-local num_allocations
__global__ void sa_test_kernel(SATestInput input, int* num_allocations)
{
    unsigned int local_thread_id
        = celeritas::KernelParamCalculator::thread_id().get();
    if (local_thread_id >= input.num_threads)
        return;

    num_allocations[local_thread_id] = 0;

    StackAllocator allocate(input.sa_view);
    for (int i = 0; i < input.num_iters; ++i)
    {
        void* new_data = allocate(input.alloc_size);
        if (new_data)
        {
            ++num_allocations[local_thread_id];
        }
    }
}

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Run on device and return allocation count.
SATestOutput sa_run(SATestInput input)
{
    thrust::device_vector<int> local_allocations(input.num_threads);

    celeritas::KernelParamCalculator calc_launch_params;
    auto params = calc_launch_params(input.num_threads);
    sa_test_kernel<<<params.grid_size, params.block_size>>>(
        input, raw_pointer_cast(local_allocations.data()));
    CELER_CUDA_CALL(cudaDeviceSynchronize());

    SATestOutput result;
    result.num_allocations = thrust::reduce(local_allocations.begin(),
                                            local_allocations.end(),
                                            0,
                                            thrust::plus<int>());
    return result;
}

//---------------------------------------------------------------------------//
} // namespace celeritas_test
