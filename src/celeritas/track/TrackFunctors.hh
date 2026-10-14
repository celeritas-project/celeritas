//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/TrackFunctors.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/sys/ThreadId.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// CONDITIONS
//---------------------------------------------------------------------------//
/*!
 * Condition for ConditionalTrackExecutor for active, non-errored tracks.
 */
struct AppliesValid
{
    template<class T>
    CELER_FUNCTION bool operator()(T const& track) const
    {
        return is_track_valid(track.sim().status());
    }
};

//---------------------------------------------------------------------------//
/*!
 * Apply only to tracks with the given post-step action ID.
 */
struct IsStepActionEqual
{
    ActionId action;

    template<class T>
    CELER_FUNCTION bool operator()(T const& track) const
    {
        return track.sim().post_step_action() == this->action;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Apply only to tracks with the given along-step action ID.
 */
struct IsAlongStepActionEqual
{
    ActionId action;

    template<class T>
    CELER_FUNCTION bool operator()(T const& track) const
    {
        CELER_EXPECT(AppliesValid{}(track)
                     == static_cast<bool>(track.sim().along_step_action()));
        return track.sim().along_step_action() == this->action;
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
