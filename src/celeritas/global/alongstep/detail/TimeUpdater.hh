//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/alongstep/detail/TimeUpdater.hh
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
    auto particle = track.make_particle_view();
    real_type speed = native_value_from(particle.speed());
    CELER_ASSERT(speed >= 0);
    if (speed > 0)
    {
        // For very small energies (< numeric_limits<real_type>::epsilon)
        // the calculated speed can be zero.
        auto sim = track.make_sim_view();
        real_type delta_time = sim.step_length() / speed;
        sim.add_time(delta_time);
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
