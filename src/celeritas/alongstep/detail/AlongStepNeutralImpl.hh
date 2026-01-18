//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/alongstep/detail/AlongStepNeutralImpl.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/global/CoreTrackView.hh"

#include "ElossApplier.hh"  // IWYU pragma: associated
#include "LinearTrackPropagator.hh"  // IWYU pragma: associated
#include "MscApplier.hh"  // IWYU pragma: associated
#include "MscStepLimitApplier.hh"  // IWYU pragma: associated
#include "PropagationApplier.hh"  // IWYU pragma: associated
#include "TimeUpdater.hh"  // IWYU pragma: associated
#include "TrackUpdater.hh"  // IWYU pragma: associated

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Helper class for along-step kernel when no MSC is in use.
 */
struct NoMsc
{
    //! MSC never applies to the current track
    CELER_FUNCTION bool
    is_applicable(CoreTrackView const&, real_type step) const
    {
        CELER_ASSERT(step > 0);
        return false;
    }

    //! No updates needed to the physical and geometric step lengths
    CELER_FUNCTION void limit_step(CoreTrackView const&) const {}

    //! MSC is never applied
    CELER_FUNCTION void apply_step(CoreTrackView const&) const {}
};

//---------------------------------------------------------------------------//
/*!
 * Class that returns zero energy loss.
 */
struct NoELoss
{
    //! No energy loss
    CELER_FUNCTION auto operator()(CoreTrackView const&) const -> decltype(auto)
    {
        return zero_quantity();
    }
};

//---------------------------------------------------------------------------//
/*!
 * Perform the along-step action using helper functions.
 */
struct AlongStepNeutralExecutor
{
    inline CELER_FUNCTION void operator()(CoreTrackView& track);

    NoMsc msc;
    LinearTrackPropagator propagate_track;
    NoELoss eloss;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
CELER_FUNCTION void AlongStepNeutralExecutor::operator()(CoreTrackView& track)
{
    MscStepLimitApplier{msc}(track);
    PropagationApplier{propagate_track}(track);
    MscApplier{msc}(track);
    TimeUpdater{}(track);
    ElossApplier{eloss}(track);
    TrackUpdater{}(track);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
