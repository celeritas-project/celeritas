//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/SmearRoughnessExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/math/ArrayOperators.hh"
#include "celeritas/optical/CoreTrackView.hh"
#include "celeritas/optical/surface/SmearRoughnessSampler.hh"
#include "celeritas/optical/surface/SurfacePhysicsUtils.hh"

#include "SmearRoughnessData.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Brief class description.
 *
 * Optional detailed class description, and possibly example usage:
 * \code
    SmearRoughnessExecutor ...;
   \endcode
 */
struct SmearRoughnessExecutor
{
    NativeCRef<SmearRoughnessData> data;

    inline CELER_FUNCTION void operator()(CoreTrackView& track) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 */
CELER_FUNCTION void
SmearRoughnessExecutor::operator()(CoreTrackView& track) const
{
    auto s_physics = track.surface_physics();

    auto dir = s_physics.traversal_direction(track.geometry().dir());
    CELER_ASSERT(!s_physics.is_exiting(dir));

    Real3 normal = s_physics.global_normal();
    if (dir == SubsurfaceDirection::reverse)
    {
        normal = -normal;
    }

    auto rng = track.rng();

    auto phys_surface_id = s_physics.subsurface_interface(dir);
    auto facet_normal = SmearRoughnessSampler{
        normal,
        data.roughness[s_physics
                           .surface_physics_map(SurfacePhysicsOrder::roughness,
                                                phys_surface_id)
                           .internal_surface_id()]}(rng);

    s_physics.facet_normal(facet_normal);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
