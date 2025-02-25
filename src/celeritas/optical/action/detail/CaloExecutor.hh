//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/action/detail/AlongStepExecutor.hh
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
 * Complete end-of-step activity for a track.
 *
 * - Update track time
 * - Update number of steps
 * - Update remaining MFPs to interaction
 */
struct CaloExecutor
{
    inline CELER_FUNCTION void operator()(CoreTrackView& track);
};

//---------------------------------------------------------------------------//
CELER_FUNCTION void CaloExecutor::operator()(CoreTrackView& track)
{
    auto sim = track.sim();
    // If track previously killed in step, don't contribute
    if (sim.status() == TrackStatus::killed)
    {
        return;
    }

    // If track is alive, check for detection
    if (sim.status() == TrackStatus::alive)
    {
        // Extract track geometry
        // auto geo = track.geometry();

        // check for track geometry in optical detector list

        // If found, print energy, print volume, and kill track
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas