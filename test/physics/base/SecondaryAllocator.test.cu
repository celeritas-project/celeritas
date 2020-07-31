//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SecondaryAllocator.test.cu
//---------------------------------------------------------------------------//
#include "SecondaryAllocator.test.hh"

#include <cstdint>
#include <thrust/device_vector.h>
#include "base/KernelParamCalculator.cuda.hh"
#include "physics/base/SecondaryAllocatorView.hh"

using thrust::raw_pointer_cast;

namespace celeritas_test
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//

__global__ void sa_test_kernel(SATestInput input, SATestOutput* output)
{
    auto thread_idx = celeritas::KernelParamCalculator::thread_id().get();
    if (thread_idx >= input.num_threads)
        return;

    SecondaryAllocatorView allocate(input.sa_view);
    for (int i = 0; i < input.num_iters; ++i)
    {
        Secondary* secondaries = allocate(input.alloc_size);
        if (!secondaries)
        {
            continue;
        }

        atomicAdd(&output->num_allocations, input.alloc_size);
        // Check that all secondaries are initialized correctly
        for (int j = 0; j < input.alloc_size; ++j)
        {
            if (secondaries[j].def_id)
            {
                atomicAdd(&output->num_errors, 1);
            }

            // Initialize the secondary
            secondaries[j].def_id = ParticleDefId{thread_idx % 4};
            for (int ax = 0; ax < 3; ++ax)
            {
                secondaries[j].direction[ax] = 0;
            }
            secondaries[j].direction[thread_idx % 3] = 1;
            secondaries[j].energy = 1 + 10 * real_type(thread_idx);
        }
        static_assert(sizeof(void*) == sizeof(SATestOutput::ull_int),
                      "Wrong pointer size");
        atomicMax(&output->last_secondary_address,
                  reinterpret_cast<SATestOutput::ull_int>(secondaries));
    }

    // Do a max on the total in-kernel size, which *might* be under
    // modification by atomics!
    atomicMax(&output->max_size, static_cast<int>(*input.sa_view.size));
}

__global__ void sa_post_test_kernel(SATestInput input, SATestOutput* output)
{
    auto thread_id = celeritas::KernelParamCalculator::thread_id();

    const SecondaryAllocatorView allocate(input.sa_view);
    if (thread_id == ThreadId{0})
    {
        output->view_size = allocate.secondaries().size();
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
    CELER_CUDA_CHECK_ERROR();

    // Access secondaries after the first kernel completed
    sa_post_test_kernel<<<params.grid_size, params.block_size>>>(
        input, raw_pointer_cast(out.data()));
    CELER_CUDA_CHECK_ERROR();

    // Copy data back to host
    thrust::host_vector<SATestOutput> host_result = out;
    return host_result.front();
}

//---------------------------------------------------------------------------//
} // namespace celeritas_test
