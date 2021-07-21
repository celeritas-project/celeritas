//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GeoTrackView.test.cu
//---------------------------------------------------------------------------//
#include "geometry/GeoTrackView.hh"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "base/KernelParamCalculator.cuda.hh"

#include "Geo.test.hh"

using thrust::raw_pointer_cast;
using namespace celeritas;

namespace celeritas_test
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//

__global__ void vgg_test_kernel(const GeoParamsCRefDevice params,
                                const GeoStateRefDevice   state,
                                const VGGTestInit*        start,
                                const int                 max_segments,
                                VolumeId*                 ids,
                                double*                   distances)
{
    CELER_EXPECT(params && state);

    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (tid.get() >= state.size())
        return;

    GeoTrackView geo(params, state, tid);
    geo = start[tid.get()];

    for (int seg = 0; seg < max_segments; ++seg)
    {
        if (geo.is_outside())
            break;

        // Move next step
        real_type dist = geo.move_next_step();

        // Save current ID and distance to travel
        ids[tid.get() * max_segments + seg]       = geo.volume_id();
        distances[tid.get() * max_segments + seg] = dist;
    }
}

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Run on device and return results
VGGTestOutput vgg_test(VGGTestInput input)
{
    CELER_EXPECT(input.params);
    CELER_EXPECT(input.state);
    CELER_EXPECT(input.init.size() == input.state.size());
    CELER_EXPECT(input.max_segments > 0);

    // Temporary device data for kernel
    thrust::device_vector<VGGTestInit> init(input.init.begin(),
                                            input.init.end());
    thrust::device_vector<VolumeId> ids(input.init.size() * input.max_segments);
    thrust::device_vector<double>   distances(ids.size(), -1.0);

    // Run kernel
    static const celeritas::KernelParamCalculator calc_launch_params(
        vgg_test_kernel, "vgg_test");
    auto params = calc_launch_params(init.size());
    vgg_test_kernel<<<params.grid_size, params.block_size>>>(
        input.params,
        input.state,
        raw_pointer_cast(init.data()),
        input.max_segments,
        raw_pointer_cast(ids.data()),
        raw_pointer_cast(distances.data()));
    CELER_CUDA_CHECK_ERROR();
    CELER_CUDA_CALL(cudaDeviceSynchronize());

    // Copy result back to CPU
    VGGTestOutput result;
    for (auto id : thrust::host_vector<VolumeId>(ids))
    {
        result.ids.push_back(id ? static_cast<int>(id.get()) : -1);
    }
    result.distances.resize(distances.size());
    thrust::copy(distances.begin(), distances.end(), result.distances.begin());

    return result;
}

//---------------------------------------------------------------------------//
} // namespace celeritas_test
