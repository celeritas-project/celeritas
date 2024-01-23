//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-rasterizer/RDemoRunner.cc
//---------------------------------------------------------------------------//
#include "RDemoRunner.hh"

#include "corecel/cont/Range.hh"
#include "corecel/data/CollectionStateStore.hh"
#include "corecel/io/ColorUtils.hh"
#include "corecel/io/Logger.hh"
#include "corecel/sys/Stopwatch.hh"
#include "celeritas/geo/GeoParams.hh"

#include "ImageTrackView.hh"
#include "RDemoKernel.hh"

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Construct with image parameters
 */
RDemoRunner::RDemoRunner(SPConstGeo geometry)
    : geo_params_(std::move(geometry))
{
    CELER_EXPECT(geo_params_);
}

//---------------------------------------------------------------------------//
/*!
 * Trace an image.
 */
void RDemoRunner::operator()(ImageStore* image, int ntimes) const
{
    CELER_EXPECT(image);

    CollectionStateStore<GeoStateData, MemSpace::device> geo_state(
        geo_params_->host_ref(), image->dims()[0]);

    CELER_LOG(status) << "Tracing geometry";
    // do it ntimes+1 as first one tends to be a warm-up run (slightly longer)
    double sum = 0, time = 0;
    for (int i = 0; i <= ntimes; ++i)
    {
        Stopwatch get_time;
        trace(geo_params_->device_ref(),
              geo_state.ref(),
              image->device_interface());
        time = get_time();
        CELER_LOG(info) << color_code('x') << "Elapsed " << i << ": " << time
                        << " s" << color_code(' ');
        if (i > 0)
        {
            sum += time;
        }
    }
    if (ntimes > 0)
    {
        CELER_LOG(info) << color_code('x')
                        << "\tAverage time: " << sum / ntimes << " s"
                        << color_code(' ');
    }
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
