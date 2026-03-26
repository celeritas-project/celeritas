//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/g4/StateDependent.hh
//---------------------------------------------------------------------------//
#pragma once

#include <functional>
#include <G4VStateDependent.hh>

#include "corecel/sys/ThreadId.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Receive notifications when the Geant4 application state changes.
 *
 * This wrapper calls a user-provided function for the given worker stream
 * whenever the simulation transitions between Geant4 application states.
 */
class StateDependent : public G4VStateDependent
{
  public:
    //!@{
    //! \name Type aliases
    using AppState = G4ApplicationState;
    using LocalStateChangeFunc
        = std::function<void(StreamId, AppState, AppState)>;
    //!@}

  public:
    // Construct with a stream ID and state-change callback
    StateDependent(StreamId sid, LocalStateChangeFunc cb);

    // Invoke the callback when the Geant4 state changes
    G4bool Notify(G4ApplicationState state) final;

  private:
    StreamId local_stream_;
    LocalStateChangeFunc cb_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
