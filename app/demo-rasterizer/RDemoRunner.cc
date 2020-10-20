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
#include "geometry/GeoParams.hh"
#include "geometry/GeoStateStore.hh"
#include "ImageTrackView.hh"
#include "RDemoKernel.hh"

using namespace celeritas;
using std::cerr;
using std::endl;

namespace demo_rasterizer
{
//---------------------------------------------------------------------------//
/*!
 * Construct with image parameters
 */
RDemoRunner::RDemoRunner(SPConstGeo geometry)
    : geo_params_(std::move(geometry))
{
    REQUIRE(geo_params_);
}

//---------------------------------------------------------------------------//
/*!
 * Trace an image.
 */
void RDemoRunner::operator()(ImageStore* image) const
{
    REQUIRE(image);

    GeoStateStore geo_state(*geo_params_, image->dims()[0]);

    cerr << "::: Tracing geometry..." << std::flush;
    Stopwatch get_time;
    trace(geo_params_->device_pointers(),
          geo_state.device_pointers(),
          image->device_interface());
    cerr << color_code('x') << " (" << get_time() << " s)" << color_code(' ')
         << endl;
}

//---------------------------------------------------------------------------//
} // namespace demo_rasterizer
