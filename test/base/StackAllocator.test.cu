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
#include "base/StackAllocator.hh"

using thrust::raw_pointer_cast;

namespace celeritas_test
{
using StackAllocatorMock = celeritas::StackAllocator<MockSecondary>;

//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//
namespace
{
__global__ void sa_test_kernel(SATestInput const input, SATestOutput* output)
{
    auto thread_idx = celeritas::KernelParamCalculator::thread_id().get();
    if (thread_idx >= input.num_threads)
        return;

    StackAllocatorMock allocate(input.sa_pointers);
    for (int i = 0; i < input.num_iters; ++i)
    {
        MockSecondary* secondaries = allocate(input.alloc_size);
        if (!secondaries)
        {
            continue;
        }

        celeritas::atomic_add(&output->num_allocations, input.alloc_size);
        for (int j = 0; j < input.alloc_size; ++j)
        {
            if (secondaries[j].mock_id != -1)
            {
                // Initialization failed (in-place new not called)
                celeritas::atomic_add(&output->num_errors, 1);
            }

            // Initialize the secondary
            secondaries[j].mock_id = thread_idx;
        }
        static_assert(sizeof(void*) == sizeof(celeritas::ull_int),
                      "Wrong pointer size");
        celeritas::atomic_max(
            &output->last_secondary_address,
            reinterpret_cast<celeritas::ull_int>(secondaries));
    }
}

__global__ void
sa_post_test_kernel(SATestInput const input, SATestOutput* output)
{
    auto thread_id = celeritas::KernelParamCalculator::thread_id();

    const StackAllocatorMock allocate(input.sa_pointers);
    if (thread_id == celeritas::ThreadId{0})
    {
        output->view_size = allocate.get().size();
    }
}

__global__ void sa_clear_kernel(SATestInput const input)
{
    auto thread_id = celeritas::KernelParamCalculator::thread_id();

    StackAllocatorMock allocate(input.sa_pointers);
    if (thread_id == celeritas::ThreadId{0})
    {
        allocate.clear();
    }
}
} // namespace

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Run on device and return results
SATestOutput sa_test(const SATestInput& input)
{
    // Construct and initialize output data
    thrust::device_vector<SATestOutput> out(1);

    static const celeritas::KernelParamCalculator calc_launch_params(
        sa_test_kernel, "sa_test");
    auto params = calc_launch_params(input.num_threads);
    sa_test_kernel<<<params.grid_size, params.block_size>>>(
        input, raw_pointer_cast(out.data()));
    CELER_CUDA_CHECK_ERROR();
    CELER_CUDA_CALL(cudaDeviceSynchronize());

    // Access secondaries after the first kernel completed
    static const celeritas::KernelParamCalculator calc_post_params(
        sa_post_test_kernel, "sa_post_test");
    params = calc_launch_params(input.num_threads);
    sa_post_test_kernel<<<params.grid_size, params.block_size>>>(
        input, raw_pointer_cast(out.data()));
    CELER_CUDA_CHECK_ERROR();
    CELER_CUDA_CALL(cudaDeviceSynchronize());

    // Copy data back to host
    thrust::host_vector<SATestOutput> host_result = out;
    return host_result.front();
}

//---------------------------------------------------------------------------//
//! Clear secondaries, only a single thread needed
void sa_clear(const SATestInput& input)
{
    static const celeritas::KernelParamCalculator calc_launch_params(
        sa_clear_kernel, "sa_clear", 32);
    auto params = calc_launch_params(1);
    sa_clear_kernel<<<params.grid_size, params.block_size>>>(input);
    CELER_CUDA_CHECK_ERROR();
    CELER_CUDA_CALL(cudaDeviceSynchronize());
}

//---------------------------------------------------------------------------//
} // namespace celeritas_test
