//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RDemoKernel.cu
//---------------------------------------------------------------------------//
#include "RDemoKernel.hh"

#include <cmath>

#include "base/Assert.hh"
#include "base/KernelParamCalculator.device.hh"
#include "geometry/GeoTrackView.hh"

#include "ImageTrackView.hh"

using namespace celeritas;
using namespace demo_rasterizer;

namespace demo_rasterizer
{
namespace
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//

__device__ int geo_id(const GeoTrackView& geo)
{
    if (geo.is_outside())
        return -1;
    return geo.volume_id().get();
}

__global__ void trace_kernel(const GeoParamsCRefDevice geo_params,
                             const GeoStateRefDevice   geo_state,
                             const ImageData           image_state)
{
    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (tid.get() >= image_state.dims[0])
        return;

    ImageTrackView image(image_state, tid);
    GeoTrackView   geo(geo_params, geo_state, tid);

    // Start track at the leftmost point in the requested direction
    geo = GeoTrackInitializer{image.start_pos(), image.start_dir()};

    int cur_id = geo_id(geo);

    // Track along each pixel
    for (unsigned int i = 0; i < image_state.dims[1]; ++i)
    {
        real_type pix_dist      = image_state.pixel_width;
        real_type max_dist      = 0;
        int       max_id        = cur_id;
        int       abort_counter = 32; // max number of crossings per pixel

        auto next = geo.find_next_step(pix_dist);
        while (next.boundary && pix_dist > 0)
        {
            CELER_ASSERT(next.distance <= pix_dist);
            // Move to geometry boundary
            pix_dist -= next.distance;

            if (max_id == cur_id)
            {
                max_dist += next.distance;
            }
            else if (next.distance > max_dist)
            {
                max_dist = next.distance;
                max_id   = cur_id;
            }

            // Cross surface and update post-crossing ID
            geo.move_to_boundary();
            geo.cross_boundary();
            cur_id = geo_id(geo);

            if (--abort_counter == 0)
            {
                // Reinitialize at end of pixel
                Real3 new_pos = image.start_pos();
                celeritas::axpy((i + 1) * image_state.pixel_width,
                                image.start_dir(),
                                &new_pos);
                geo      = GeoTrackInitializer{new_pos, image.start_dir()};
                pix_dist = 0;
            }
            if (pix_dist > 0)
            {
                // Next movement is to end of geo or pixel
                next = geo.find_next_step(pix_dist);
            }
        }

        if (pix_dist > 0)
        {
            // Move to pixel boundary
            geo.move_internal(pix_dist);
            if (pix_dist > max_dist)
            {
                max_dist = pix_dist;
                max_id   = cur_id;
            }
        }
        image.set_pixel(i, max_id);
    }
}
} // namespace

//---------------------------------------------------------------------------//
// KERNEL INTERFACE
//---------------------------------------------------------------------------//
void trace(const GeoParamsCRefDevice& geo_params,
           const GeoStateRefDevice&   geo_state,
           const ImageData&           image)
{
    CELER_EXPECT(image);

    static const KernelParamCalculator calc_kernel_params(trace_kernel,
                                                          "trace");

    auto params = calc_kernel_params(image.dims[0]);
    trace_kernel<<<params.grid_size, params.block_size>>>(
        geo_params, geo_state, image);
    CELER_DEVICE_CHECK_ERROR();

    CELER_DEVICE_CALL_PREFIX(DeviceSynchronize());
}

//---------------------------------------------------------------------------//
} // namespace demo_rasterizer
