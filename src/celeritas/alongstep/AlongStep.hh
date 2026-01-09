//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/alongstep/AlongStep.hh
//! \brief Along-step function and helper classes
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/global/CoreTrackView.hh"

#include "detail/ElossApplier.hh"  // IWYU pragma: associated
#include "detail/MscApplier.hh"  // IWYU pragma: associated
#include "detail/MscStepLimitApplier.hh"  // IWYU pragma: associated
#include "detail/PropagationApplier.hh"  // IWYU pragma: associated
#include "detail/TimeUpdater.hh"  // IWYU pragma: associated
#include "detail/TrackUpdater.hh"  // IWYU pragma: associated

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Perform the along-step action using helper functions.
 *
 * \todo move into ASNA.cc, only place it's used
 *
 * \tparam MH MSC helper, e.g. \c detail::NoMsc
 * \tparam TP Track propagator
 * \tparam EH Energy loss helper, e.g. \c detail::NoELoss
 */
template<class MH, class TP, class EH>
struct AlongStep
{
    inline CELER_FUNCTION void operator()(CoreTrackView& track);

    MH msc;
    TP propagate_track;
    EH eloss;
};

//---------------------------------------------------------------------------//
// DEDUCTION GUIDES
//---------------------------------------------------------------------------//
template<class MH, class TP, class EH>
CELER_FUNCTION AlongStep(MH&&, TP&&, EH&&) -> AlongStep<MH, TP, EH>;

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
template<class MH, class TP, class EH>
CELER_FUNCTION void AlongStep<MH, TP, EH>::operator()(CoreTrackView& track)
{
    detail::MscStepLimitApplier{msc}(track);
    detail::PropagationApplier{propagate_track}(track);
    detail::MscApplier{msc}(track);
    detail::TimeUpdater{}(track);
    detail::ElossApplier{eloss}(track);
    detail::TrackUpdater{}(track);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
