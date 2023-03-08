//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/StackAllocator.test.cu
//---------------------------------------------------------------------------//
#include "StackAllocator.test.hh"

#include <cstdint>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "corecel/device_runtime_api.h"
#include "corecel/data/StackAllocator.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/KernelParamCalculator.device.hh"

using thrust::raw_pointer_cast;

namespace celeritas
{
namespace test
{
using StackAllocatorMock = StackAllocator<MockSecondary>;

//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//
namespace
{
__global__ void sa_test_kernel(SATestInput const input, SATestOutput* output)
{
    auto thread_idx = TrackSlotId{KernelParamCalculator::thread_id().unchecked_get()}.get();
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

        atomic_add(&output->num_allocations, input.alloc_size);
        for (int j = 0; j < input.alloc_size; ++j)
        {
            if (secondaries[j].mock_id != -1)
            {
                // Initialization failed (in-place new not called)
                atomic_add(&output->num_errors, 1);
            }

            // Initialize the secondary
            secondaries[j].mock_id = thread_idx;
        }
        static_assert(sizeof(void*) == sizeof(ull_int), "Wrong pointer size");
        atomic_max(&output->last_secondary_address,
                   reinterpret_cast<ull_int>(secondaries));
    }
}

__global__ void
sa_post_test_kernel(SATestInput const input, SATestOutput* output)
{
    auto thread_id = TrackSlotId{KernelParamCalculator::thread_id().unchecked_get()};

    const StackAllocatorMock allocate(input.sa_data);
    if (thread_id == ThreadId{0})
    {
        output->view_size = allocate.get().size();
    }
}

__global__ void sa_clear_kernel(SATestInput const input)
{
    auto thread_id = TrackSlotId{KernelParamCalculator::thread_id().unchecked_get()};

    StackAllocatorMock allocate(input.sa_data);
    if (thread_id == ThreadId{0})
    {
        allocate.clear();
    }
}
}  // namespace

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Run on device and return results
SATestOutput sa_test(SATestInput const& input)
{
    // Construct and initialize output data
    thrust::device_vector<SATestOutput> out(1);

    CELER_LAUNCH_KERNEL(sa_test,
                        device().default_block_size(),
                        input.num_threads,
                        input,
                        raw_pointer_cast(out.data()));
    CELER_DEVICE_CALL_PREFIX(DeviceSynchronize());

    // Access secondaries after the first kernel completed
    CELER_LAUNCH_KERNEL(sa_post_test,
                        device().default_block_size(),
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
void sa_clear(SATestInput const& input)
{
    CELER_LAUNCH_KERNEL(sa_clear, device().threads_per_warp(), 1, input);
    CELER_DEVICE_CALL_PREFIX(DeviceSynchronize());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
