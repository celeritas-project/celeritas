//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/g4/SteppingAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <functional>
#include <G4UserSteppingAction.hh>

#include "corecel/sys/ThreadId.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Dispatch Geant4 step data to a thread-local callback function.
 *
 * This stepping action wraps a user-provided function, calling it for each
 * step with the current worker stream ID and the step data.
 */
class SteppingAction final : public G4UserSteppingAction
{
  public:
    //!@{
    //! \name Type aliases
    using FuncLocalStep = std::function<void(StreamId, G4Step const&)>;
    //!@}

  public:
    // Construct with a stream ID and per-step callback
    explicit SteppingAction(StreamId sid, FuncLocalStep f);

    // Dispatch a step to the user callback
    void UserSteppingAction(G4Step const* step) final;

  private:
    StreamId sid_;
    FuncLocalStep callback_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
