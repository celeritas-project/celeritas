//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GCheckKernel.cu
//---------------------------------------------------------------------------//
#include "GCheckKernel.hh"

#include "base/Assert.hh"
#include "base/KernelParamCalculator.cuda.hh"
#include "geometry/GeoInterface.hh"
#include "geometry/GeoTrackView.hh"
#include "geometry/LinearPropagator.hh"
#include <cmath>

using namespace celeritas;

namespace geo_check
{
namespace
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//

__global__ void gcheck_kernel(const GeoParamsPointers   geo_params,
                              const GeoStatePointers    geo_state,
                              const GeoStateInitializer init,
                              int                       ntrks,
                              int                       max_steps)
{
    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (tid.get() >= ntrks)
        return;

    GeoTrackView     geo(geo_params, geo_state, tid);
    LinearPropagator propagate(&geo);

    // Start track at the leftmost point in the requested direction
    geo = GeoStateInitializer{init};

    real_type geo_dist = geo.next_step();

    // Track along each pixel
    for (int istep = 0; istep < max_steps; ++istep)
    {
        if (geo.is_outside())
            break;

        // Save current ID and distance to travel
        auto step = propagate();
        printf("tid=%i step=%i: volid=%i, dist=%f\n",
               tid.get(),
               istep,
               (geo.is_outside() ? -1 : (int)step.volume.get()),
               step.distance);
    }
}
} // namespace

//---------------------------------------------------------------------------//
// KERNEL INTERFACE
//---------------------------------------------------------------------------//
/*!
 *  Run tracking on the GPU
 */
void run_gpu(const GeoParamsPointers&   geo_params,
             const GeoStatePointers&    geo_state,
             const GeoStateInitializer& init,
             int                        max_steps)
{
    // CELER_EXPECT(init);

    // static const KernelParamCalculator calc_kernel_params(trace_kernel,
    //                                                       "trace");

    // auto params = calc_kernel_params(init);
    gcheck_kernel<<<1, 1>>>(geo_params, geo_state, init, 1, max_steps);
    CELER_CUDA_CHECK_ERROR();

    CELER_CUDA_CALL(cudaDeviceSynchronize());
}

//---------------------------------------------------------------------------//
} // namespace geo_check
