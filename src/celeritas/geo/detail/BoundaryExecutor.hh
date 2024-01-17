//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/geo/detail/BoundaryExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/Types.hh"
#include "celeritas/geo/GeoMaterialView.hh"
#include "celeritas/geo/GeoTrackView.hh"
#include "celeritas/global/CoreTrackView.hh"
#include "celeritas/mat/MaterialTrackView.hh"
#include "celeritas/track/SimTrackView.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
struct BoundaryExecutor
{
    inline CELER_FUNCTION void
    operator()(celeritas::CoreTrackView const& track);
};

//---------------------------------------------------------------------------//
/*!
 * Cross a geometry boundary.
 *
 * \pre The track must have already been physically moved to the correct point
 * on the boundary.
 */
CELER_FUNCTION void
BoundaryExecutor::operator()(celeritas::CoreTrackView const& track)
{
    CELER_EXPECT([track] {
        auto sim = track.make_sim_view();
        return sim.post_step_action() == track.boundary_action()
               && sim.status() == TrackStatus::alive;
    }());

    auto geo = track.make_geo_view();
    CELER_EXPECT(geo.is_on_boundary());

    // Particle entered a new volume before reaching the interaction point
    geo.cross_boundary();
    if (!geo.is_outside())
    {
        // Update the material in the new region
        auto geo_mat = track.make_geo_material_view();
        auto matid = geo_mat.material_id(geo.volume_id());
        CELER_ASSERT(matid);
        auto mat = track.make_material_view();
        mat = {matid};

        CELER_ENSURE(geo.is_on_boundary());
    }
    else
    {
        auto sim = track.make_sim_view();
        sim.status(TrackStatus::killed);
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
