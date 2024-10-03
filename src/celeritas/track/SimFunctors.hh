//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/SimFunctors.hh
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
    CELER_FUNCTION bool operator()(T const& sim) const
    {
        return is_track_valid(sim.status());
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
    CELER_FUNCTION bool operator()(T const& sim) const
    {
        return sim.post_step_action() == this->action;
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
    CELER_FUNCTION bool operator()(T const& sim) const
    {
        CELER_EXPECT(AppliesValid{}(sim)
                     == static_cast<bool>(sim.along_step_action()));
        return sim.along_step_action() == this->action;
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
