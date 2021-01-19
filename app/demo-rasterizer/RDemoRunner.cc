//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RDemoRunner.cc
//---------------------------------------------------------------------------//
#include "RDemoRunner.hh"

#include "base/Range.hh"
#include "base/Stopwatch.hh"
#include "base/ColorUtils.hh"
#include "comm/Logger.hh"
#include "geometry/GeoParams.hh"
#include "geometry/GeoStateStore.hh"
#include "ImageTrackView.hh"
#include "RDemoKernel.hh"

using namespace celeritas;

namespace demo_rasterizer
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
void RDemoRunner::operator()(ImageStore* image) const
{
    CELER_EXPECT(image);

    GeoStateStore geo_state(*geo_params_, image->dims()[0]);

    CELER_LOG(status) << "Tracing geometry";
    Stopwatch get_time;
    trace(geo_params_->device_pointers(),
          geo_state.device_pointers(),
          image->device_interface());
    CELER_LOG(diagnostic) << color_code('x') << "... " << get_time() << " s"
                          << color_code(' ');
}

//---------------------------------------------------------------------------//
} // namespace demo_rasterizer
