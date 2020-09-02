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

namespace
{
//---------------------------------------------------------------------------//
/*!
 * Initialize the geometry state on host.
 */
void initialize_geo_state(const GeoParams&                      geo_params,
                          const demo_rasterizer::ImagePointers& shared,
                          GeoStateStore*                        state)
{
    REQUIRE(state && state->size() == shared.dims[0]);
    unsigned int       num_y_pixels = shared.dims[0];
    std::vector<Real3> start_pos(num_y_pixels);
    std::vector<Real3> start_dir(num_y_pixels);

    // Loop over thread index (vertical pixel)
    for (auto thread_idx : celeritas::range(num_y_pixels))
    {
        demo_rasterizer::ImageTrackView view(shared, ThreadId(thread_idx));

        start_pos[thread_idx] = view.start_pos();
        start_dir[thread_idx] = view.start_dir();
    }

    state->initialize(geo_params, start_pos, start_dir);
}

//---------------------------------------------------------------------------//
} // namespace

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

    Stopwatch get_time;
    cerr << "::: Initializing raytrace..." << std::flush;
    GeoStateStore geo_state(*geo_params_, image->dims()[0]);
    initialize_geo_state(*geo_params_, image->host_interface(), &geo_state);
    cerr << color_code('x') << " (" << get_time() << " s)" << color_code(' ')
         << endl;

    cerr << "::: Tracing geometry..." << std::flush;
    get_time = {};
    trace(geo_params_->device_pointers(),
          geo_state.device_pointers(),
          image->device_interface());
    cerr << color_code('x') << " (" << get_time() << " s)" << color_code(' ')
         << endl;
}

//---------------------------------------------------------------------------//
} // namespace demo_rasterizer
