//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/G4VecgeomConversion.test.cu
//---------------------------------------------------------------------------//
#include "G4VecgeomConversion.test.hh"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "corecel/sys/KernelParamCalculator.device.hh"
#include "celeritas/ext/VecgeomTrackView.hh"

using thrust::raw_pointer_cast;
using namespace celeritas;

namespace celeritas_test
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//

__global__ void g4vgconv_test_kernel(const GeoParamsCRefDevice  params,
                                     const GeoStateRefDevice    state,
                                     const GeoTrackInitializer* start,
                                     const int                  max_segments,
                                     int*                       ids,
                                     double*                    distances)
{
    CELER_EXPECT(params && state);

    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (tid.get() >= state.size())
        return;

    VecgeomTrackView geo(params, state, tid);
    geo = start[tid.get()];

    for (int seg = 0; seg < max_segments; ++seg)
    {
        // Move next step
        auto next = geo.find_next_step();
        if (next.boundary)
        {
            geo.move_to_boundary();
            geo.cross_boundary();
        }

        // Save current ID and distance travelled
        ids[tid.get() * max_segments + seg]
            = (geo.is_outside()
                   ? -2
                   : static_cast<int>(geo.volume_id().unchecked_get()));
        distances[tid.get() * max_segments + seg] = next.distance;

        if (geo.is_outside())
            break;
    }
}

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Run on device and return results
G4VGConvTestOutput g4vgconv_test(G4VGConvTestInput input)
{
    CELER_EXPECT(input.params);
    CELER_EXPECT(input.state);
    CELER_EXPECT(input.init.size() == input.state.size());
    CELER_EXPECT(input.max_segments > 0);

    // Temporary device data for kernel
    thrust::device_vector<GeoTrackInitializer> init(input.init.begin(),
                                                    input.init.end());
    thrust::device_vector<int> ids(input.init.size() * input.max_segments, -3);
    thrust::device_vector<double> distances(ids.size(), -3.0);

    // Run kernel
    static const celeritas::KernelParamCalculator calc_launch_params(
        g4vgconv_test_kernel, "g4vgconv_test");
    auto params = calc_launch_params(init.size());
    g4vgconv_test_kernel<<<params.blocks_per_grid, params.threads_per_block>>>(
        input.params,
        input.state,
        raw_pointer_cast(init.data()),
        input.max_segments,
        raw_pointer_cast(ids.data()),
        raw_pointer_cast(distances.data()));
    CELER_CUDA_CALL(cudaPeekAtLastError());
    CELER_CUDA_CALL(cudaDeviceSynchronize());

    // Copy result back to CPU
    G4VGConvTestOutput result;
    result.ids.resize(ids.size());
    thrust::copy(ids.begin(), ids.end(), result.ids.begin());
    result.distances.resize(distances.size());
    thrust::copy(distances.begin(), distances.end(), result.distances.begin());

    return result;
}

//---------------------------------------------------------------------------//
} // namespace celeritas_test
