//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file StackAllocator.test.cu
//---------------------------------------------------------------------------//
#include "StackAllocator.test.hh"

#include <cstdint>
#include <thrust/device_vector.h>
#include "base/KernelParamCalculator.cuda.hh"
#include "base/StackAllocatorView.hh"

using thrust::raw_pointer_cast;

namespace celeritas_test
{
using StackAllocatorViewMock = celeritas::StackAllocatorView<MockSecondary>;

//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//

__global__ void sa_test_kernel(SATestInput input, SATestOutput* output)
{
    auto thread_idx = celeritas::KernelParamCalculator::thread_id().get();
    if (thread_idx >= input.num_threads)
        return;

    StackAllocatorViewMock allocate(input.sa_pointers);
    for (int i = 0; i < input.num_iters; ++i)
    {
        MockSecondary* secondaries = allocate(input.alloc_size);
        if (!secondaries)
        {
            continue;
        }

        atomicAdd(&output->num_allocations, input.alloc_size);
        for (int j = 0; j < input.alloc_size; ++j)
        {
            if (secondaries[j].def_id != -1)
            {
                // Initialization failed (in-place new not called)
                atomicAdd(&output->num_errors, 1);
            }

            // Initialize the secondary
            secondaries[j].def_id = thread_idx;
        }
        static_assert(sizeof(void*) == sizeof(celeritas::ull_int),
                      "Wrong pointer size");
        atomicMax(&output->last_secondary_address,
                  reinterpret_cast<celeritas::ull_int>(secondaries));
    }

    // Do a max on the total in-kernel size, which *might* be under
    // modification by atomics!
    atomicMax(&output->max_size, static_cast<int>(*input.sa_pointers.size));
}

__global__ void sa_post_test_kernel(SATestInput input, SATestOutput* output)
{
    auto thread_id = celeritas::KernelParamCalculator::thread_id();

    const StackAllocatorViewMock allocate(input.sa_pointers);
    if (thread_id == celeritas::ThreadId{0})
    {
        output->view_size = allocate.get().size();
    }
}

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Run on device and return results
SATestOutput sa_test(SATestInput input)
{
    // Construct and initialize output data
    thrust::device_vector<SATestOutput> out(1);

    celeritas::KernelParamCalculator calc_launch_params;
    auto params = calc_launch_params(input.num_threads);
    sa_test_kernel<<<params.grid_size, params.block_size>>>(
        input, raw_pointer_cast(out.data()));
    CELER_CUDA_CALL(cudaDeviceSynchronize());

    // Access secondaries after the first kernel completed
    sa_post_test_kernel<<<params.grid_size, params.block_size>>>(
        input, raw_pointer_cast(out.data()));
    CELER_CUDA_CALL(cudaDeviceSynchronize());

    // Copy data back to host
    thrust::host_vector<SATestOutput> host_result = out;
    return host_result.front();
}

//---------------------------------------------------------------------------//
} // namespace celeritas_test
