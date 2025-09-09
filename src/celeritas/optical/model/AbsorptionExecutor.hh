//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/model/AbsorptionExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "celeritas/optical/CoreTrackView.hh"
#include "celeritas/optical/interactor/AbsorptionInteractor.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
struct AbsorptionExecutor
{
    inline CELER_FUNCTION Interaction operator()(CoreTrackView const&);
};

//---------------------------------------------------------------------------//
/*!
 * Sample optical absorption interaction from the current track.
 */
CELER_FUNCTION Interaction AbsorptionExecutor::operator()(CoreTrackView const&)
{
    return AbsorptionInteractor{}();
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
