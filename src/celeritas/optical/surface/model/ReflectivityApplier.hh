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
 * ReflectivityResult.
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
ReflectivityApplier::operator()(CoreTrackView const& track) const
{
    // Sample reflectivity
    switch (this->sample_reflectivity(track))
    {
        case ReflectivityResult::absorb:
            track.sim().status(TrackStatus::killed);
            break;
        case ReflectivityResult::interact:
            // Do nothing if photon should undergo surface interaction
            break;
        default:
            // Catch pass-through for future changes
            CELER_ASSERT_UNREACHABLE();
    }
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
