//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/PolishedRoughnessExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/math/ArrayOperators.hh"
#include "celeritas/optical/CoreTrackView.hh"
#include "celeritas/optical/surface/SurfacePhysicsUtils.hh"

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
    PolishedRoughnessExecutor ...;
   \endcode
 */
struct PolishedRoughnessExecutor
{
    inline CELER_FUNCTION void operator()(CoreTrackView& track) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 */
CELER_FUNCTION void
PolishedRoughnessExecutor::operator()(CoreTrackView& track) const
{
    auto s_physics = track.surface_physics();

    auto dir = s_physics.traversal_direction(track.geometry().dir());
    CELER_ASSERT(!s_physics.is_exiting(dir));

    Real3 normal = s_physics.global_normal();
    if (dir == SubsurfaceDirection::reverse)
    {
        normal = -normal;
    }

    s_physics.facet_normal(normal);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
