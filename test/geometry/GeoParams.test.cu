//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GeoParams.test.cu
//---------------------------------------------------------------------------//

#include <thrust/device_vector.h>
#include "base/KernelParamCalculator.cuda.hh"
#include "GeoParams.test.hh"

using thrust::raw_pointer_cast;

namespace celeritas_test
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//

__global__ void
gp_test_kernel(const GeoParamsPointers shared, const int max_segments)
{
    auto tid = celeritas::KernelParamCalculator::thread_id();
    // if (tid.get() >= size) {
    if (tid.get() >= 1)
    {
        return;
    }
    // TODO: check geometry on device
    // We could do some random point locations?
    REQUIRE(true);
}

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Run on device and return results
GPTestOutput gp_test(GPTestInput input)
{
    REQUIRE(input.shared);
    REQUIRE(input.max_segments > 0);

    // Temporary device data for kernel
    // thrust::device_vector<VGGTestInit> init(input.init.begin(),
    //                                         input.init.end());
    // thrust::device_vector<VolumeId> ids(input.init.size() *
    // input.max_segments);
    //                                         input.init.end());

    // Run kernel
    celeritas::KernelParamCalculator calc_launch_params;
    // auto params = calc_launch_params(init.size());
    auto params = calc_launch_params(1);
    gp_test_kernel<<<params.grid_size, params.block_size>>>(
        input.shared, input.max_segments);
    CELER_CUDA_CALL(cudaDeviceSynchronize());

    // Copy result back to CPU
    GPTestOutput result;
    // for (auto id : thrust::host_vector<VolumeId>(ids))
    // {
    //     result.ids.push_back(id ? static_cast<int>(id.get()) : -1);
    // }
    // result.distances.resize(distances.size());
    // thrust::copy(distances.begin(), distances.end(),
    // result.distances.begin());

    return result;
}

//---------------------------------------------------------------------------//
} // namespace celeritas_test
