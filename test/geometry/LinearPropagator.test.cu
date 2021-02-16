//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LinearPropagator.test.cu
//---------------------------------------------------------------------------//
#include "LinearPropagator.test.hh"

#include <thrust/device_vector.h>
#include "base/KernelParamCalculator.cuda.hh"
#include "geometry/GeoTrackView.hh"
#include "geometry/LinearPropagator.hh"

using thrust::raw_pointer_cast;

namespace celeritas_test
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//

__global__ void linProp_test_kernel(const GeoParamsPointers shared,
                                    const GeoStatePointers  state,
                                    const int               size,
                                    const LinPropTestInit*  start,
                                    const int               max_segments,
                                    VolumeId*               ids,
                                    double*                 distances)
{
    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (tid.get() >= size)
        return;

    GeoTrackView geo(shared, state, tid);
    geo = start[tid.get()];

    LinearPropagator propagate(&geo);
    for (int seg = 0; seg < max_segments; ++seg)
    {
        if (geo.is_outside())
            break;
        geo.find_next_step();

        // Save current ID and distance to travel
        ids[tid.get() * max_segments + seg]       = geo.volume_id();
        distances[tid.get() * max_segments + seg] = geo.next_step();

        // Move next step
        propagate();
    }
}

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Run on device and return results
LinPropTestOutput linProp_test(LinPropTestInput input)
{
    CELER_EXPECT(input.shared);
    CELER_EXPECT(input.state);
    CELER_EXPECT(input.init.size() == input.state.size);
    CELER_EXPECT(input.max_segments > 0);

    // Temporary device data for kernel
    thrust::device_vector<LinPropTestInit> init(input.init.begin(),
                                                input.init.end());
    thrust::device_vector<VolumeId> ids(input.init.size() * input.max_segments);
    thrust::device_vector<double>   distances(ids.size(), -1.0);

    // Run kernel
    static const celeritas::KernelParamCalculator calc_launch_params(
        linProp_test_kernel, "linProp_test");
    auto                             params = calc_launch_params(init.size());
    linProp_test_kernel<<<params.grid_size, params.block_size>>>(
        input.shared,
        input.state,
        init.size(),
        raw_pointer_cast(init.data()),
        input.max_segments,
        raw_pointer_cast(ids.data()),
        raw_pointer_cast(distances.data()));
    CELER_CUDA_CHECK_ERROR();
    CELER_CUDA_CALL(cudaDeviceSynchronize());

    // Copy result back to CPU
    LinPropTestOutput result;
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
