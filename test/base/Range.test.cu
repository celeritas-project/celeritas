//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Range.test.cu
//---------------------------------------------------------------------------//
#include "Range.test.hh"

#include <thrust/device_vector.h>
#include "base/Assert.hh"
#include "base/Range.hh"

using celeritas::range;

namespace celeritas_test
{
__global__ void
rangedev_test_kernel(int a, int* x, int* y, int* z, unsigned int n)
{
    // grid stride loop
    for (auto i : range(blockIdx.x * blockDim.x + threadIdx.x, n)
                      .step(gridDim.x * blockDim.x))
    {
        z[i] = a * x[i] + y[i];
    }
}

RangeTestOutput rangedev_test(RangeTestInput input)
{
    REQUIRE(input.x.size() == input.y.size());

    // Local device vectors for working data
    thrust::device_vector<int> x_dev(input.x.begin(), input.x.end());
    thrust::device_vector<int> y_dev(input.y.begin(), input.y.end());
    thrust::device_vector<int> z_dev(input.x.size(), 0);

    // Test kernel
    rangedev_test_kernel<<<input.num_threads, input.num_blocks>>>(
        input.a,
        thrust::raw_pointer_cast(x_dev.data()),
        thrust::raw_pointer_cast(y_dev.data()),
        thrust::raw_pointer_cast(z_dev.data()),
        z_dev.size());

    // Copy result back to CPU
    RangeTestOutput result;
    result.z.assign(z_dev.size(), 0);
    thrust::copy(z_dev.begin(), z_dev.end(), result.z.begin());

    return result;
}

//---------------------------------------------------------------------------//
} // namespace celeritas_test