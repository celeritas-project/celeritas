//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/ReflectivityApplier.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/optical/CoreTrackView.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Wrap a reflectivity executor and apply it to a track.
 *
 * The functor \c F must take a \c CoreTrackView and return a \c
 * ReflectivityAction.
 */
template<class F>
struct ReflectivityApplier
{
    F sample_reflectivity;

    inline CELER_FUNCTION void operator()(CoreTrackView const&) const;
};

//---------------------------------------------------------------------------//
// DEDUCTION GUIDES
//---------------------------------------------------------------------------//
template<class F>
CELER_FUNCTION ReflectivityApplier(F&&) -> ReflectivityApplier<F>;

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Apply sampled reflectivity result to the track.
 */
template<class F>
CELER_FUNCTION void
ReflectivityApplier<F>::operator()(CoreTrackView const& track) const
{
    // Sample reflectivity and set it
    auto action = this->sample_reflectivity(track);

    auto s_phys = track.surface_physics();
    s_phys.reflectivity_action(action);

    if (action == ReflectivityAction::absorb)
    {
        // Mark particle as killed
        track.sim().status(TrackStatus::killed);
    }
    else if (action == ReflectivityAction::transmit)
    {
        // Move across the boundary
        auto traverse = s_phys.traversal();
        traverse.cross_interface(traverse.dir());
        if (traverse.is_exiting())
        {
            // End boundary crossing if exiting
            track.sim().post_step_action(s_phys.scalars().post_boundary_action);
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
