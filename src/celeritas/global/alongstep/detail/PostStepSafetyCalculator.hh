//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/alongstep/detail/PostStepSafetyCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/global/CoreTrackView.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Calculate safety distance at the end of a step.
 *
 * TODO: this is currently *only* for MSC.
 */
template<class MH>
struct PostStepSafetyCalculator
{
    inline CELER_FUNCTION void operator()(CoreTrackView const& track);

    MH msc;
};

//---------------------------------------------------------------------------//
// DEDUCTION GUIDES
//---------------------------------------------------------------------------//
template<class MH>
CELER_FUNCTION PostStepSafetyCalculator(MH&&)->PostStepSafetyCalculator<MH>;

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//

template<class MH>
CELER_FUNCTION void
PostStepSafetyCalculator<MH>::operator()(CoreTrackView const& track)
{
    CELER_EXPECT(track.make_sim_view().status() == TrackStatus::alive);

    auto cache = track.make_safety_cache_view();
    if (cache.is_on_boundary())
    {
        // No safety update
        return;
    }

    real_type min_safety = msc.safety_post(track);
    if (min_safety > 0)
    {
        cache.find_safety(min_safety);
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
