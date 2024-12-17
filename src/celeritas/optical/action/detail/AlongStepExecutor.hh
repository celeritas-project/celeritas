//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
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
struct AlongStepExecutor
{
    inline CELER_FUNCTION void operator()(CoreTrackView& track);
};

//---------------------------------------------------------------------------//
CELER_FUNCTION void AlongStepExecutor::operator()(CoreTrackView& track)
{
    auto sim = track.sim();

    // Update time
    sim.add_time(sim.step_length() / constants::c_light);

    CELER_ASSERT(sim.status() == TrackStatus::alive);
    CELER_ASSERT(sim.step_length() > 0);
    CELER_ASSERT(sim.post_step_action());

    // TODO: update step count and check max step cut
    // TODO: reduce MFP by step * xs
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas
