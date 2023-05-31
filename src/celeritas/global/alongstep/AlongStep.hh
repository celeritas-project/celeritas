//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/alongstep/AlongStep.hh
//! \brief Along-step function and helper classes
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/global/CoreTrackView.hh"

#include "detail/AlongStepImpl.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Perform the along-step action using helper functions.
 *
 * \tparam MH MSC helper, e.g. \c detail::NoMsc
 * \tparam MP Propagator factory, e.g. \c detail::LinearPropagatorFactory
 * \tparam EH Energy loss helper, e.g. \c detail::TrackNoEloss
 */
template<class MH, class MP, class EH>
struct AlongStep
{
    inline CELER_FUNCTION void operator()(CoreTrackView const& track);

    MH msc;
    MP make_propagator;
    EH eloss;
};

//---------------------------------------------------------------------------//
// DEDUCTION GUIDES
//---------------------------------------------------------------------------//
template<class MH, class MP, class EH>
CELER_FUNCTION AlongStep(MH&&, MP&&, EH&&)->AlongStep<MH, MP, EH>;

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
template<class MH, class MP, class EH>
CELER_FUNCTION void
AlongStep<MH, MP, EH>::operator()(CoreTrackView const& track)
{
    detail::MscStepLimitApplier{msc}(track);
    detail::PropagationApplier{make_propagator}(track);
    detail::MscApplier{msc}(track);
    detail::TimeUpdater{}(track);
    detail::ElossApplier{eloss}(track);
    detail::TrackUpdater{}(track);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
