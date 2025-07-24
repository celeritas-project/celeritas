//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/PostBoundaryExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/optical/CoreTrackView.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 */
struct PostBoundaryExecutor
{
    CELER_FUNCTION void operator()(CoreTrackView&) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Finalize the track's boundary crossing.
 */
CELER_FUNCTION void PostBoundaryExecutor::operator()(CoreTrackView& track) const
{
    CELER_EXPECT(track.is_crossing_boundary());

    auto surface_physics = track.surface_physics();

    if (surface_physics.in_pre_volume())
    {
        // re-entrant in pre-volume
    }
    else
    {
        // entrant in post-volume
    }

    surface_physics.reset();
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
