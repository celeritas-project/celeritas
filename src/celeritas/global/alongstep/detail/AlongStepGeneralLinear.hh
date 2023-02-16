//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/alongstep/detail/AlongStepGeneralLinear.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/em/data/FluctuationData.hh"
#include "celeritas/em/data/UrbanMscData.hh"

#include "AlongStepNeutral.hh"
#include "FluctELoss.hh"
#include "UrbanMsc.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Implementation of the "along step" action with MSC and eloss fluctuation.
 */
inline CELER_FUNCTION void
along_step_general_linear(const NativeCRef<UrbanMscData>& msc,
                          NoData,
                          const NativeCRef<FluctuationData>& fluct,
                          CoreTrackView const&               track)
{
    return along_step(
        UrbanMsc{msc}, LinearPropagatorFactory{}, FluctELoss{fluct}, track);
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
