//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/vg/Vecgeom.test.cu
//---------------------------------------------------------------------------//
#include "Vecgeom.test.hh"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "corecel/sys/KernelParamCalculator.device.hh"
#include "geocel/vg/VecgeomTrackView.hh"

using thrust::raw_pointer_cast;

namespace celeritas
{
namespace test
{
namespace
{
//---------------------------------------------------------------------------//
__global__ void vgg_test_kernel(DeviceCRef<VecgeomParamsData> const params,
                                DeviceRef<VecgeomStateData> const state,
                                GeoTrackInitializer const* start,
                                int const max_segments,
                                int* ids,
                                double* distances)
{
    CELER_EXPECT(params && state);

    auto tid = TrackSlotId{KernelParamCalculator::thread_id().unchecked_get()};
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
}  // namespace

//! Run on device and return results
VGGTestOutput vgg_test(VGGTestInput const& input)
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
    CELER_LAUNCH_KERNEL(vgg_test,
                        init.size(),
                        0,
                        input.params,
                        input.state,
                        raw_pointer_cast(init.data()),
                        input.max_segments,
                        raw_pointer_cast(ids.data()),
                        raw_pointer_cast(distances.data()));

    CELER_DEVICE_API_CALL(DeviceSynchronize());

    // Copy result back to CPU
    VGGTestOutput result;
    result.ids.resize(ids.size());
    thrust::copy(ids.begin(), ids.end(), result.ids.begin());
    result.distances.resize(distances.size());
    thrust::copy(distances.begin(), distances.end(), result.distances.begin());

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
