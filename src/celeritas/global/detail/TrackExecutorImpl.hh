//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/detail/TrackExecutorImpl.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/sys/ThreadId.hh"
#include "celeritas/Types.hh"
#include "celeritas/track/SimTrackView.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
// CONDITIONS
//---------------------------------------------------------------------------//
/*!
 * Condition for ConditionalTrackExecutor for active tracks.
 */
struct AppliesActive
{
    CELER_FUNCTION bool operator()(SimTrackView const& sim) const
    {
        return sim.status() != TrackStatus::inactive;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Apply only to tracks with the given post-step action ID.
 */
struct IsStepActionEqual
{
    ActionId action;

    CELER_FUNCTION bool operator()(SimTrackView const& sim) const
    {
        return sim.step_limit().action == this->action;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Apply only to tracks with the given along-step action ID.
 */
struct IsAlongStepActionEqual
{
    ActionId action;

    CELER_FUNCTION bool operator()(SimTrackView const& sim) const
    {
        CELER_EXPECT(AppliesActive{}(sim)
                     == static_cast<bool>(sim.along_step_action()));
        return sim.along_step_action() == this->action;
    }
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
