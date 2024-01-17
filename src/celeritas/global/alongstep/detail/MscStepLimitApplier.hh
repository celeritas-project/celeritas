//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/alongstep/detail/MscStepLimitApplier.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/global/CoreTrackView.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Apply MSC step limiters.
 *
 * TODO: think about integrating this into the pre-step sequence. Maybe the
 * geo/phys path transformation would be best suited to the \c
 * apply_propagation step?
 */
template<class MH>
struct MscStepLimitApplier
{
    inline CELER_FUNCTION void operator()(CoreTrackView const& track);

    MH msc;
};

//---------------------------------------------------------------------------//
// DEDUCTION GUIDES
//---------------------------------------------------------------------------//

template<class MH>
CELER_FUNCTION MscStepLimitApplier(MH&&) -> MscStepLimitApplier<MH>;

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//

template<class MH>
CELER_FUNCTION void
MscStepLimitApplier<MH>::operator()(CoreTrackView const& track)
{
    if (msc.is_applicable(track, track.make_sim_view().step_length()))
    {
        // Apply MSC step limiters and transform "physical" step (with MSC) to
        // "geometrical" step (smooth curve)
        msc.limit_step(track);

        auto step_view = track.make_physics_step_view();
        CELER_ASSERT(step_view.msc_step().geom_path > 0);
    }
    else
    {
        // TODO: hack flag for saving "use_msc"
        auto step_view = track.make_physics_step_view();
        step_view.msc_step().geom_path = 0;
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
