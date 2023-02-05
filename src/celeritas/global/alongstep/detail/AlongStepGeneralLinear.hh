//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/alongstep/detail/AlongStepGeneralLinear.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/em/data/FluctuationData.hh"
#include "celeritas/em/data/UrbanMscData.hh"
#include "celeritas/em/msc/UrbanMsc.hh"

#include "AlongStepNeutral.hh"
#include "EnergyLossFluctApplier.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Implementation of the "along step" action with MSC and eloss fluctuation.
 */
inline CELER_FUNCTION void
along_step_general_linear(NativeCRef<UrbanMscData> const& msc,
                          NoData,
                          NativeCRef<FluctuationData> const& fluct,
                          CoreTrackView const& track)
{
    return along_step(UrbanMsc{msc},
                      LinearPropagatorFactory{},
                      EnergyLossFluctApplier{fluct},
                      track);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
