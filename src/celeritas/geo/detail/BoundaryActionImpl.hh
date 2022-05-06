//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/geo/detail/BoundaryActionImpl.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/global/CoreTrackView.hh"

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
inline CELER_FUNCTION void boundary_track(celeritas::CoreTrackView const& track)
{
    auto sim = track.make_sim_view();
    if (sim.step_limit().action != track.boundary_action())
    {
        // Not undergoing a boundary crossing
        return;
    }
    CELER_ASSERT(sim.status() == TrackStatus::alive);

    auto geo = track.make_geo_view();

    // Particle entered a new volume before reaching the interaction point
    geo.cross_boundary();
    if (!geo.is_outside())
    {
        // Update the material in the new region
        auto geo_mat = track.make_geo_material_view();
        auto matid   = geo_mat.material_id(geo.volume_id());
        CELER_ASSERT(matid);
        auto mat = track.make_material_view();
        mat      = {matid};
    }
    else
    {
        auto sim = track.make_sim_view();
        sim.status(TrackStatus::killed);
    }
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
