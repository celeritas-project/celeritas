//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/action/detail/PropagateExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/Types.hh"
#include "celeritas/optical/CoreTrackView.hh"
#include "celeritas/optical/SimTrackView.hh"

namespace celeritas
{
namespace optical
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Move a track to the next interaction or geometry boundary.
 *
 * This should only apply to alive tracks.
 */
struct PropagateExecutor
{
    inline CELER_FUNCTION void operator()(CoreTrackView& track);
};

//---------------------------------------------------------------------------//
CELER_FUNCTION void PropagateExecutor::operator()(CoreTrackView& track)
{
    auto&& sim = track.sim();
    CELER_ASSERT(sim.status() == TrackStatus::alive);

    // Propagate up to the physics distance
    real_type step = sim.step_length();
    CELER_ASSERT(step > 0);

    auto&& geo = track.geometry();
    Propagation p = geo.find_next_step(step);
    if (p.boundary)
    {
        geo.move_to_boundary();
        sim.step_length(p.distance);
        sim.post_step_action(track.init_boundary_action());
    }
    else
    {
        CELER_ASSERT(step == p.distance);
        geo.move_internal(step);
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas
