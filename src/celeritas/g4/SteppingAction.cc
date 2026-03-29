//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/g4/SteppingAction.cc
//---------------------------------------------------------------------------//
#include "SteppingAction.hh"

#include <G4Step.hh>

#include "corecel/Assert.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with a stream ID and per-step callback.
 */
SteppingAction::SteppingAction(StreamId sid, FuncLocalStep f)
    : sid_{sid}, callback_{std::move(f)}
{
    CELER_EXPECT(callback_);
}

//---------------------------------------------------------------------------//
/*!
 * Dispatch a step to the user callback.
 */
void SteppingAction::UserSteppingAction(G4Step const* step)
{
    CELER_EXPECT(step);
    CELER_EXPECT(sid_);
    callback_(sid_, *step);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
