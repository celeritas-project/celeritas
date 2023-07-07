//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/alongstep/detail/PreStepSafetyCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/global/CoreTrackView.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Calculate safety distance at the beginning of a step.
 *
 * TODO: this is currently *only* for MSC. We will want to extend this to
 * accommodate the physics step so that we don't have to do any geometry
 * interaction when far from boundaries.
 */
template<class MH>
struct PreStepSafetyCalculator
{
    inline CELER_FUNCTION void operator()(CoreTrackView const& track);

    MH msc;
};

//---------------------------------------------------------------------------//
// DEDUCTION GUIDES
//---------------------------------------------------------------------------//
template<class MH>
CELER_FUNCTION PreStepSafetyCalculator(MH&&)->PreStepSafetyCalculator<MH>;

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//

template<class MH>
CELER_FUNCTION void
PreStepSafetyCalculator<MH>::operator()(CoreTrackView const& track)
{
    CELER_ASSERT(track.make_sim_view().status() == TrackStatus::alive);

    auto geo = track.make_safety_cache_view();
    if (geo.is_on_boundary())
    {
        // Safety is zero
        return;
    }

    if (!msc.is_applicable(track, track.make_sim_view().step_limit().step))
    {
        return;
    }

    real_type min_safety = msc.safety_pre(track);
    if (min_safety > 0)
    {
        // Calculate and cache safety
        geo.find_safety(min_safety);
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
