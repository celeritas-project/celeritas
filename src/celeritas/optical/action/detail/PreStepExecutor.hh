//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/action/detail/PreStepExecutor.hh
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
 * Set up the beginning of a physics step.
 */
struct PreStepExecutor
{
    inline CELER_FUNCTION void operator()(CoreTrackView const& track);
};

//---------------------------------------------------------------------------//
CELER_FUNCTION void PreStepExecutor::operator()(CoreTrackView const& track)
{
    auto sim = track.sim();
    if (sim.status() == TrackStatus::killed)
    {
        // Deactivate tracks killed in previous loop
        sim.status(TrackStatus::inactive);
    }
    if (sim.status() == TrackStatus::inactive)
    {
        // Clear step limit and actions for an empty track slot
        sim.reset_step_limit();
        return;
    }

    if (CELER_UNLIKELY(sim.status() == TrackStatus::errored))
    {
        // Failed during initialization: don't calculate step limits
        return;
    }

    CELER_ASSERT(sim.status() == TrackStatus::initializing
                 || sim.status() == TrackStatus::alive);

    if (sim.status() == TrackStatus::initializing)
    {
        sim.reset_step_limit();
        sim.status(TrackStatus::alive);
    }

    // TODO: reset secondaries
    // TODO: calculate step limit
    CELER_ENSURE(sim.step_length() > 0);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas
