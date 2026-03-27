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
 *
 * The transitions are encoded as `GeantStateChange` values according to
 * the following table (previous -> requested):
 *
 * | Previous State         | Requested State        | Change        |
 * |------------------------|------------------------|---------------|
 * | `G4State_PreInit`      | `G4State_Init`         | `initialize`  |
 * | `G4State_Idle`         | `G4State_Init`         | `initialize`  |
 * | `G4State_Idle`         | `G4State_GeomClosed`   | `begin_run`   |
 * | `G4State_GeomClosed`   | `G4State_EventProc`    | `begin_event` |
 * | `G4State_EventProc`    | `G4State_GeomClosed`   | `end_event`   |
 * | `G4State_GeomClosed`   | `G4State_Idle`         | `end_run`     |
 * | *other*                | *other*                | `unknown`     |
 */
class StateDependent : public G4VStateDependent
{
  public:
    //!@{
    //! \name Type aliases
    using AppState = G4ApplicationState;
    //! Encodes a meaningful state transition for user callbacks.
    enum class GeantStateChange
    {
        initialize,  //!< G4State_PreInit -> G4State_Init or G4State_Idle ->
                     //!< G4State_Init
        begin_run,  //!< G4State_Idle -> G4State_GeomClosed
        begin_event,  //!< G4State_GeomClosed -> G4State_EventProc
        end_event,  //!< G4State_EventProc -> G4State_GeomClosed
        end_run,  //!< G4State_GeomClosed -> G4State_Idle
        unknown  //!< Any other or unexpected transition
    };

    // Callback receives a stream id and the encoded `GeantStateChange`.
    using LocalStateChangeFunc
        = std::function<void(StreamId, GeantStateChange)>;
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
