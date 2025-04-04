//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/geo/detail/BoundaryExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/Types.hh"
#include "celeritas/global/CoreTrackView.hh"
#include "celeritas/mat/MaterialTrackView.hh"
#include "celeritas/track/SimTrackView.hh"

#include "../GeoMaterialView.hh"
#include "../GeoTrackView.hh"

#if !CELER_DEVICE_COMPILE
#    include "corecel/io/Logger.hh"
#endif

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Cross a geometry boundary.
 *
 * \pre The track must have already been physically moved to the correct point
 * on the boundary.
 */
struct BoundaryExecutor
{
    inline CELER_FUNCTION void operator()(celeritas::CoreTrackView& track);
};

//---------------------------------------------------------------------------//
CELER_FUNCTION void
BoundaryExecutor::operator()(celeritas::CoreTrackView& track)
{
    CELER_EXPECT([track] {
        auto sim = track.sim();
        return sim.post_step_action() == track.boundary_action()
               && sim.status() == TrackStatus::alive;
    }());

    auto geo = track.geometry();
    CELER_EXPECT(geo.is_on_boundary());

    // Particle entered a new volume before reaching the interaction point
    geo.cross_boundary();
    if (CELER_UNLIKELY(geo.failed()))
    {
        track.apply_errored();
        return;
    }
    else if (!geo.is_outside())
    {
        // Update the material in the new region
        auto geo_mat = track.geo_material();
        auto matid = geo_mat.material_id(geo.volume_id());
        if (CELER_UNLIKELY(!matid))
        {
#if !CELER_DEVICE_COMPILE
            CELER_LOG_LOCAL(error)
                << R"(Track entered a volume without an associated material)";
#endif
            track.apply_errored();
            return;
        }
        auto mat = track.material();
        mat = {matid};

        CELER_ENSURE(geo.is_on_boundary());
    }
    else
    {
        auto sim = track.sim();
        sim.status(TrackStatus::killed);
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
