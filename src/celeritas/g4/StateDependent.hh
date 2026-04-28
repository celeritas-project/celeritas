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
 * Encode a meaningful Geant4 state transition for user callbacks.
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
 * | *any*                  | `G4State_Quit`         | `end_program` |
 * | *any*                  | `G4State_Abort`        | `end_program` |
 * | *other*                | *other*                | `unknown`     |
 *
 * \par Notes:
 *
 * - \c end_program is called by the G4RunManager and G4RunManagerKernel
 *   destructor during normal execution (via \c G4State_Quit)
 * - \c end_program is called by \c G4Exception when a fatal error occurred
 *   (via \c G4State_Abort)
 */
enum class GeantStateChange
{
    initialize,
    begin_run,
    begin_event,
    end_event,
    end_run,
    end_program,
    unknown,
    size_,
};

// Callback receives a stream id and the encoded `GeantStateChange`.
using LocalGeantStateChangeFunc
    = std::function<void(StreamId, GeantStateChange)>;

//---------------------------------------------------------------------------//
/*!
 * Receive notifications when the Geant4 application state changes.
 *
 * This thread-local wrapper calls a shared user-provided function for the
 * given worker stream whenever the simulation transitions between Geant4
 * application states.
 *
 * \warning The Geant4 base class constructor/destructor calls
 * register/deregister(this) on the thread-local G4StateManager, which means
 * it's not very safe to destroy this on a thread other than the one that
 * created it. The StreamId accessor can be used to check on what thread it was
 * created. You should probably put an instance of this in your RunAction or
 * TrackManager in order to prevent the world from exploding.
 */
class StateDependent final : public G4VStateDependent
{
  public:
    //!@{
    //! \name Type aliases
    using LocalStateChangeFunc = LocalGeantStateChangeFunc;
    //!@}

  public:
    // Construct locally with state-change callback
    explicit StateDependent(LocalStateChangeFunc cb);

    // Invoke the callback when the Geant4 state changes
    G4bool Notify(G4ApplicationState state) final;

    //! Stream that created this state dependent
    StreamId local_stream() const { return local_stream_; }

  private:
    StreamId local_stream_;
    LocalStateChangeFunc cb_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
