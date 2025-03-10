//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/alongstep/detail/MscApplier.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/global/CoreTrackView.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Apply multiple scattering.
 *
 * This does three key things:
 * - Replaces the "geometrical" step (continuous) with the "physical" step
 *   (including multiple scattering)
 * - Likely changes the direction of the track
 * - Possibly displaces the particle
 */
template<class MH>
struct MscApplier
{
    inline CELER_FUNCTION void operator()(CoreTrackView const& track);

    MH msc;
};

//---------------------------------------------------------------------------//
// DEDUCTION GUIDES
//---------------------------------------------------------------------------//
template<class MH>
CELER_FUNCTION MscApplier(MH&&) -> MscApplier<MH>;

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
template<class MH>
CELER_FUNCTION void MscApplier<MH>::operator()(CoreTrackView const& track)
{
    if (track.sim().status() != TrackStatus::alive)
    {
        // Active track killed during propagation or erroneous: don't apply MSC
        return;
    }

    if (track.physics_step().msc_step().geom_path > 0)
    {
        // Scatter the track and transform the "geometrical" step back to
        // "physical" step
        msc.apply_step(track);
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
