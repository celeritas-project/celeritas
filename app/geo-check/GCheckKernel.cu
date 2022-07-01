//---------------------------------*-Cudac-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GCheckKernel.cu
//---------------------------------------------------------------------------//
#include "corecel/Assert.hh"
#include "corecel/sys/KernelParamCalculator.device.hh"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "GCheckKernel.hh"
#include "celeritas/geo/GeoData.hh"
#include "celeritas/geo/GeoTrackView.hh"
#include "celeritas/field/LinearPropagator.hh"
#include "corecel/data/CollectionStateStore.hh"

using namespace celeritas;
using thrust::raw_pointer_cast;

namespace geo_check
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//

__device__ int geo_physid(const GeoTrackView& geo)
{
    if (geo.is_outside())
        return -1;
    return geo.volume_physid().get();
}

__global__ void gcheck_kernel(const GeoParamsCRefDevice  params,
                              const GeoStateRefDevice    state,
                              const GeoTrackInitializer* init,
                              int                        max_steps,
                              int*                       ids,
                              double*                    distances)
{
    CELER_EXPECT(params && state);
    CELER_EXPECT(max_steps > 0);

    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (tid.get() >= state.size())
        return;

    celeritas::GeoTrackView     geo(params, state, tid);
    celeritas::LinearPropagator propagate(&geo);

    // Start track at the leftmost point in the requested direction
    geo = init[tid.get()];

    // Track along detector
    int istep = 0;
    do
    {
        // Propagate Save next-volume ID and distance to travel
        auto step        = propagate();
        if (step.boundary) geo.cross_boundary();
        ids[istep]       = geo.volume_physid().get();
        distances[istep] = step.distance;
        ++istep;
    } while (!geo.is_outside() && istep < max_steps);
}

//---------------------------------------------------------------------------//
// KERNEL INTERFACE
//---------------------------------------------------------------------------//
/*!
 *  Run tracking on the GPU
 */
GCheckOutput run_gpu(GCheckInput input)
{
    CELER_EXPECT(input.params);
    CELER_EXPECT(input.state);
    CELER_EXPECT(input.max_steps > 0);

    // Temporary device data for kernel
    thrust::device_vector<GeoTrackInitializer> tracks(input.init.begin(),
                                                      input.init.end());
    thrust::device_vector<int>    ids(input.init.size() * input.max_steps, -1);
    thrust::device_vector<double> distances(ids.size(), -1.0);

    // static const KernelParamCalculator calc_kernel_params(gcheck_kernel,
    // 							  "run_gpu");
    // auto launch_params = calc_launch_params(init.size());

    gcheck_kernel<<<1, 1>>>(input.params,
                            input.state,
                            raw_pointer_cast(tracks.data()),
                            input.max_steps,
                            raw_pointer_cast(ids.data()),
                            raw_pointer_cast(distances.data()));

    CELER_DEVICE_CHECK_ERROR();
    CELER_CUDA_CALL(cudaDeviceSynchronize());

    // Copy result back to CPU
    GCheckOutput result;

    // figure out how many valid steps returned
    size_type    nstep = 0;
    for (auto id : thrust::host_vector<int>(ids))
    {
        if (id < 0) break;
        ++nstep;
    }
    // Return exact vector size for proper comparison with CPU
    result.ids.resize(nstep);
    thrust::copy(ids.begin(), ids.begin() + nstep, result.ids.begin());

    result.distances.resize(nstep);
    thrust::copy(
        distances.begin(), distances.begin() + nstep, result.distances.begin());

    return result;
}

//---------------------------------------------------------------------------//
} // namespace geo_check
