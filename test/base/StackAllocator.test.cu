//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file StackAllocator.test.cu
//---------------------------------------------------------------------------//
#include "StackAllocator.test.hh"

#include <cstdint>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "base/device_runtime_api.h"
#include "base/KernelParamCalculator.device.hh"
#include "base/StackAllocator.hh"
#include "comm/Device.hh"

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

    StackAllocatorMock allocate(input.sa_data);
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

    const StackAllocatorMock allocate(input.sa_data);
    if (thread_id == celeritas::ThreadId{0})
    {
        output->view_size = allocate.get().size();
    }
}

__global__ void sa_clear_kernel(SATestInput const input)
{
    auto thread_id = celeritas::KernelParamCalculator::thread_id();

    StackAllocatorMock allocate(input.sa_data);
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

    CELER_LAUNCH_KERNEL(sa_test,
                        celeritas::device().default_block_size(),
                        input.num_threads,
                        input,
                        raw_pointer_cast(out.data()));
    CELER_DEVICE_CALL_PREFIX(DeviceSynchronize());

    // Access secondaries after the first kernel completed
    CELER_LAUNCH_KERNEL(sa_post_test,
                        celeritas::device().default_block_size(),
                        input.num_threads,
                        input,
                        raw_pointer_cast(out.data()));
    CELER_DEVICE_CALL_PREFIX(DeviceSynchronize());

    // Copy data back to host
    thrust::host_vector<SATestOutput> host_result = out;
    return host_result.front();
}

//---------------------------------------------------------------------------//
//! Clear secondaries, only a single thread needed
void sa_clear(const SATestInput& input)
{
    CELER_LAUNCH_KERNEL(
        sa_clear, celeritas::device().threads_per_warp(), 1, input);
    CELER_DEVICE_CALL_PREFIX(DeviceSynchronize());
}

//---------------------------------------------------------------------------//
} // namespace celeritas_test
