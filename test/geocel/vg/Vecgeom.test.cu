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

#include "corecel/math/NumericLimits.hh"
#include "corecel/sys/KernelParamCalculator.device.hh"
#include "geocel/vg/VecgeomTrackView.hh"
#include "geocel/vg/VecgeomTypes.hh"

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
                                double* distances,
                                double* safeties,
                                double* bounded_safeties,
                                double* small_steps)
{
    CELER_EXPECT(params && state);

    auto tid = TrackSlotId{KernelParamCalculator::thread_id().unchecked_get()};
    if (tid.get() >= state.size())
        return;

    VecgeomTrackView geo(params, state, tid);
    geo = start[tid.get()];

    // Exercise the safety path used by MSC before moving to a boundary
    safeties[tid.get()] = geo.find_safety();
    bounded_safeties[tid.get()] = geo.find_safety(real_type{1});

    // A sub-tolerance physics step must not be reported as a boundary
    auto small = geo.find_next_step(real_type{1e-20});
    small_steps[tid.get()] = small.boundary ? -small.distance : small.distance;

    for (int seg = 0; seg < max_segments; ++seg)
    {
        // Move next step
        auto next = geo.find_next_step(NumericLimits<real_type>::infinity());
        if (next.boundary)
        {
            geo.move_to_boundary(next.distance);
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
    thrust::device_vector<double> safeties(init.size(), -3.0);
    thrust::device_vector<double> bounded_safeties(init.size(), -3.0);
    thrust::device_vector<double> small_steps(init.size(), -3.0);

    // Run kernel
    CELER_LAUNCH_KERNEL(vgg_test,
                        init.size(),
                        0,
                        input.params,
                        input.state,
                        raw_pointer_cast(init.data()),
                        input.max_segments,
                        raw_pointer_cast(ids.data()),
                        raw_pointer_cast(distances.data()),
                        raw_pointer_cast(safeties.data()),
                        raw_pointer_cast(bounded_safeties.data()),
                        raw_pointer_cast(small_steps.data()));

    CELER_DEVICE_API_CALL(DeviceSynchronize());

    // Copy result back to CPU
    VGGTestOutput result;
    result.ids.resize(ids.size());
    thrust::copy(ids.begin(), ids.end(), result.ids.begin());
    result.distances.resize(distances.size());
    result.safeties.resize(safeties.size());
    result.bounded_safeties.resize(bounded_safeties.size());
    result.small_steps.resize(small_steps.size());
    thrust::copy(distances.begin(), distances.end(), result.distances.begin());
    thrust::copy(safeties.begin(), safeties.end(), result.safeties.begin());
    thrust::copy(bounded_safeties.begin(),
                 bounded_safeties.end(),
                 result.bounded_safeties.begin());
    thrust::copy(
        small_steps.begin(), small_steps.end(), result.small_steps.begin());

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
