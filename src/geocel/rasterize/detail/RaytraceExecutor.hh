//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/rasterize/detail/RaytraceExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "../ImageData.hh"
#include "../ImageLineView.hh"
#include "../Raytracer.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Get the volume ID as an integer, or -1 if outside.
 */
struct VolumeIdCalculator
{
    template<class GTV>
    CELER_FUNCTION int operator()(GTV const& geo) const
    {
        if (geo.is_outside())
            return -1;
        return static_cast<int>(geo.impl_volume_id().get());
    }
};

//---------------------------------------------------------------------------//
/*!
 * Raytrace a geometry onto an image.
 */
template<class GTV, class F>
struct RaytraceExecutor
{
    //// TYPES ////

    // TODO: Use observer pointers?
    using GeoTrackView = GTV;
    using GeoParamsRef = typename GTV::ParamsRef;
    using GeoStateRef = typename GTV::StateRef;
    using ImgParamsRef = NativeCRef<ImageParamsData>;
    using ImgStateRef = NativeRef<ImageStateData>;

    //// DATA ////

    GeoParamsRef geo_params;
    GeoStateRef geo_state;
    ImgParamsRef img_params;
    ImgStateRef img_state;

    F calc_id;

    //// FUNCTIONS ////

    // Initialize track states
    inline CELER_FUNCTION void operator()(ThreadId tid) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Trace a single line.
 */
template<class GTV, class F>
CELER_FUNCTION void RaytraceExecutor<GTV, F>::operator()(ThreadId tid) const
{
    CELER_EXPECT(tid < img_params.scalars.dims[0]);
    CELER_EXPECT(geo_state.size() == img_params.scalars.dims[0]);

    // Trace one state per vertical line
    GeoTrackView geo{geo_params, geo_state, TrackSlotId{tid.unchecked_get()}};
    ImageLineView line{img_params, img_state, tid.unchecked_get()};
    Raytracer trace(geo, calc_id, line);
    for (auto col : range(line.max_index()))
    {
        int val = trace(col);
        line.set_pixel(col, val);
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
