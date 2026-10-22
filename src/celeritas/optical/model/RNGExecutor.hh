//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/model/RayleighExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>

#include "corecel/random/engine/RanluxppRngEngine.hh"
#include "celeritas/optical/CoreTrackView.hh"
#include "celeritas/optical/interactor/RNGInteractor.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
struct RNGExecutor
{
    inline CELER_FUNCTION void operator()(CoreTrackView const&);
};

//---------------------------------------------------------------------------//
/*!
 * Sample optical Rayleigh interaction from the current track.
 */
CELER_FUNCTION void RNGExecutor::operator()(CoreTrackView const& track)
{
#if (CELERITAS_CORE_RNG == CELERITAS_CORE_RNG_RANLUXPP)
    auto rng = track.rng();
    RNGInteractor interact{};
    interact(rng);
#endif
    (void)track;  // Suppress warning
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
