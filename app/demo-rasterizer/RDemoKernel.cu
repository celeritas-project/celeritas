//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-rasterizer/RDemoKernel.cu
//---------------------------------------------------------------------------//
#include "RDemoKernel.hh"

#include <cmath>

#include "corecel/Assert.hh"
#include "corecel/sys/KernelParamCalculator.device.hh"
#include "celeritas/geo/GeoTrackView.hh"

#include "ImageTrackView.hh"

namespace celeritas
{
namespace app
{
namespace
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//

__device__ int geo_id(GeoTrackView const& geo)
{
    if (geo.is_outside())
        return -1;
    return geo.volume_id().get();
}

__global__ void trace_kernel(const GeoParamsCRefDevice geo_params,
                             const GeoStateRefDevice geo_state,
                             const ImageData image_state)
{
    auto tid = KernelParamCalculator::thread_id();
    if (tid.get() >= image_state.dims[0])
        return;

    TrackSlotId tsid{tid.unchecked_get()};
    ImageTrackView image(image_state, tsid);
    GeoTrackView geo(geo_params, geo_state, tsid);

    // Start track at the leftmost point in the requested direction
    geo = GeoTrackInitializer{image.start_pos(), image.start_dir()};

    int cur_id = geo_id(geo);

    // Track along each pixel
    for (unsigned int i = 0; i < image_state.dims[1]; ++i)
    {
        real_type pix_dist = image_state.pixel_width;
        real_type max_dist = 0;
        int max_id = cur_id;
        int abort_counter = 32;  // max number of crossings per pixel

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
                max_id = cur_id;
            }

            // Cross surface and update post-crossing ID
            geo.move_to_boundary();
            geo.cross_boundary();
            cur_id = geo_id(geo);

            if (--abort_counter == 0)
            {
                // Reinitialize at end of pixel
                Real3 new_pos = image.start_pos();
                axpy((i + 1) * image_state.pixel_width,
                     image.start_dir(),
                     &new_pos);
                geo = GeoTrackInitializer{new_pos, image.start_dir()};
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
                max_id = cur_id;
            }
        }
        image.set_pixel(i, max_id);
    }
}
}  // namespace

//---------------------------------------------------------------------------//
// KERNEL INTERFACE
//---------------------------------------------------------------------------//
void trace(GeoParamsCRefDevice const& geo_params,
           GeoStateRefDevice const& geo_state,
           ImageData const& image)
{
    CELER_EXPECT(image);

    CELER_LAUNCH_KERNEL(trace, image.dims[0], 0, geo_params, geo_state, image);

    CELER_DEVICE_CALL_PREFIX(DeviceSynchronize());
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
