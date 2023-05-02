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
inline CELER_FUNCTION void along_step(CoreTrackView const& track,
                                      MH&& msc,
                                      MP&& make_propagator,
                                      EH&& eloss)
{
    detail::apply_msc_step_limit<MH&>(track, msc);
    detail::apply_propagation<MP&>(track, make_propagator);
    detail::apply_msc<MH&>(track, msc);
    detail::update_time(track);
    detail::apply_eloss<EH&>(track, eloss);
    detail::update_track(track);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
