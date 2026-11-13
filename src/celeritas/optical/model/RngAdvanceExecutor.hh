//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/model/RngAdvanceExecutor.hh
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
    //! Advance the random number generator if using Ranlux++
    inline CELER_FUNCTION void operator()(CoreTrackView const& track)
    {
#if CELERITAS_CORE_RNG == CELERITAS_CORE_RNG_RANLUXPP
        track.rng().advance();
#endif
        CELER_DISCARD(track);
    }
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
