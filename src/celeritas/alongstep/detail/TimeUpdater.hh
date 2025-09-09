//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/alongstep/detail/TimeUpdater.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/global/CoreTrackView.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Update the lab frame time.
 */
struct TimeUpdater
{
    inline CELER_FUNCTION void operator()(CoreTrackView const& track);
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
CELER_FUNCTION void TimeUpdater::operator()(CoreTrackView const& track)
{
    auto sim = track.sim();

    // The track errored within the along-step kernel
    if (sim.status() == TrackStatus::errored)
        return;

    auto particle = track.particle();
    real_type speed = native_value_from(particle.speed());
    CELER_ASSERT(speed > 0);

    real_type delta_time = sim.step_length() / speed;
    sim.add_time(delta_time);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
