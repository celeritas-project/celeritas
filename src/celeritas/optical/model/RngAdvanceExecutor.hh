//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/model/RayleighExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/random/engine/RanluxppRngEngine.hh"
#include "celeritas/optical/CoreTrackView.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
struct RngAdvanceExecutor
{
    inline CELER_FUNCTION void operator()(CoreTrackView const&);
};

//---------------------------------------------------------------------------//
/*!
 * Sample optical Rayleigh interaction from the current track.
 */
CELER_FUNCTION void RngAdvanceExecutor::operator()(CoreTrackView const& track)
{
#if (CELERITAS_CORE_RNG == CELERITAS_CORE_RNG_RANLUXPP)
    track.rng().advance();
#endif
    (void)track;  // Suppress warning
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
