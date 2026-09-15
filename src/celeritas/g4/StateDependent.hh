//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/g4/StateDependent.hh
//---------------------------------------------------------------------------//
#pragma once

#include <functional>
#include <G4VStateDependent.hh>

#include "corecel/Macros.hh"
#include "corecel/sys/ThreadId.hh"

class G4StateManager;

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Encode a meaningful Geant4 state transition for user callbacks.
 *
 * The transitions are encoded as `GeantStateChange` values according to
 * the following table (previous -> requested):
 *
 * | Previous State         | Requested State        | Change          |
 * |------------------------|------------------------|-----------------|
 * | *none*                 | `G4State_PreInit`      | `begin_program` |
 * | `G4State_PreInit`      | `G4State_Init`         | `initialize`    |
 * | `G4State_Idle`         | `G4State_Init`         | `begin_init`    |
 * | `G4State_Init`         | `G4State_Init`         | `internal_init` |
 * | `G4State_Init`         | `G4State_Idle`         | `end_init`      |
 * | `G4State_Idle`         | `G4State_GeomClosed`   | `begin_run`     |
 * | `G4State_GeomClosed`   | `G4State_EventProc`    | `begin_event`   |
 * | `G4State_EventProc`    | `G4State_GeomClosed`   | `end_event`     |
 * | `G4State_GeomClosed`   | `G4State_Idle`         | `end_run`       |
 * | *any*                  | `G4State_Quit`         | `end_program`   |
 * | *any*                  | `G4State_Abort`        | `end_program`   |
 * | *other*                | *other*                | `unknown`       |
 *
 * \par Notes:
 *
 * - \c begin_program is called during the \c G4RunManagerKernel constructor,
 *   so it will only register if your state dependent is constructed very early
 *   and on the main thread.
 * - We ignore all but the first \c Initialize state (from "pre-init")
 * - In MT mode, \c RunManager::Initialize actually calls begin/end run for
 *   each thread, including master.
 * - \c end_program is called by the G4RunManager and G4RunManagerKernel
 *   destructor during normal execution (via \c G4State_Quit).
 * - \c end_program is called by \c G4Exception when a fatal error occurred
 *   (via \c G4State_Abort).
 */
enum class GeantStateChange
{
    begin_program,
    initialize,
    begin_init,
    internal_init,
    end_init,
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
// HELPER FUNCTIONS (HOST)
//---------------------------------------------------------------------------//

// Get a string corresponding to a Geant4 state change
char const* to_cstring(GeantStateChange);

//---------------------------------------------------------------------------//
/*!
 * Receive notifications when the Geant4 application state changes.
 *
 * This thread-local wrapper calls a shared user-provided function for the
 * given worker stream whenever the simulation transitions between Geant4
 * application states.
 *
 * \warning  The Geant4 memory semantics for this class are bonkers.
 * - The thread-local \c G4StateManager singleton keeps a pointer (via the \c
 *   G4VStateDependent base constructor) to this local instance and calls \c
 *   delete on it when it's deleted as the run manager shuts down.
 *   Therefore this class \c must be deleted before \c G4StateManager ends.
 * - The base class destructor calls the thread-local \c G4StateManager without
 *   checking for validity, so the destructor of this class \em after deleting
 *   the run manager will also crash the code.
 * - The thread-local pointer mapping means that this class \em must be
 *   deallocated on the thread in which it's created.
 *
 * To bypass the first failure path, we use \c Notify to deregister ourselves
 * when we see the run manager is about to exit or abort.
 *
 * The only truly safe way to manage memory for this class is probably to leak
 * it.
 */
class StateDependent final : public G4VStateDependent
{
  public:
    // Construct locally with state-change callback
    explicit StateDependent(LocalGeantStateChangeFunc cb);

    // Prevent move/copy due to weird base class antics
    CELER_DELETE_COPY_MOVE(StateDependent);

    // Invoke the callback when the Geant4 state changes
    G4bool Notify(G4ApplicationState state) final;

    //! Stream that created this state dependent
    StreamId local_stream() const { return local_stream_; }

  private:
    StreamId local_stream_;
    LocalGeantStateChangeFunc cb_;
    G4StateManager* manager_{nullptr};
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
