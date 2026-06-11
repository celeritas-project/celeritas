//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/g4/StateDependent.cc
//---------------------------------------------------------------------------//
#include "StateDependent.hh"

#include <memory>
#include <utility>
#include <G4StateManager.hh>
#include <G4Threading.hh>
#include <G4VStateDependent.hh>

#include "corecel/Assert.hh"
#include "corecel/io/EnumStringMapper.hh"
#include "corecel/io/Logger.hh"

#include "Threading.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with a stream ID and state-change callback.
 *
 * \note The base class performs the actual registration.
 * \note We also store a pointer to the thread-local manager that this
 * is registered with, in case we want to deregister in a thread other than
 * the one we were created with. (This might be dangerous... but so is assuming
 * we're destroyed on the same thread we're constructed in.)
 */
StateDependent::StateDependent(LocalGeantStateChangeFunc cb, Mode mode)
    : local_stream_{geant_stream()}
    , cb_{std::move(cb)}
    , manager_{G4StateManager::GetStateManager()}
    , mode_{mode}
{
    CELER_EXPECT(cb_);
    CELER_EXPECT(manager_);
    CELER_LOG_LOCAL(debug) << "Registered state dependent "
                           << static_cast<void*>(this) << " on "
                           << local_stream_;
}

//---------------------------------------------------------------------------//
/*!
 * Dispatch a state transition notification to the user callback.
 */
G4bool StateDependent::Notify(G4ApplicationState state)
{
    G4StateManager* sm = G4StateManager::GetStateManager();
    CELER_ASSERT(sm == manager_);
    G4ApplicationState prev = sm->GetPreviousState();
    // Map (previous, requested) Geant4 states to our semantic enum.
    auto change = GeantStateChange::unknown;
    switch (state)
    {
        case G4State_PreInit:
            // Constructing run kernel
            change = GeantStateChange::begin_program;
            break;
        case G4State_Init:
            if (prev == G4State_PreInit)
            {
                // First initialization: do an extra call
                this->cb_(local_stream_, GeantStateChange::initialize);
                change = GeantStateChange::begin_init;
            }
            else if (prev == G4State_Idle)
            {
                // Reinitialization
                change = GeantStateChange::begin_init;
            }
            else if (prev == G4State_Init)
            {
                // During initialization of geometry/physics
                change = GeantStateChange::internal_init;
            }
            break;
        case G4State_Idle:
            // Returning from top-level run manager call (init/beamon)
            if (prev == G4State_Init)
            {
                change = GeantStateChange::end_init;
            }
            else if (prev == G4State_GeomClosed)
            {
                change = GeantStateChange::end_run;
            }
            break;
        case G4State_GeomClosed:
            // In between events
            if (prev == G4State_Idle)
            {
                change = GeantStateChange::begin_run;
            }
            else if (prev == G4State_EventProc)
            {
                change = GeantStateChange::end_event;
            }
            break;
        case G4State_EventProc:
            // Starting an event
            if (prev == G4State_GeomClosed)
            {
                change = GeantStateChange::begin_event;
            }
            break;
        case G4State_Quit:
            [[fallthrough]];
        case G4State_Abort:
            // Tearing down the run manager or aborting
            change = GeantStateChange::end_program;
            break;
        default:
            break;
    }
    if (change == GeantStateChange::unknown)
    {
        CELER_LOG_LOCAL(debug)
            << "Unknown Geant4 state change: " << sm->GetStateString(prev)
            << "->" << sm->GetStateString(state);
    }
    else if (change == GeantStateChange::end_program)
    {
        // Deregister before exiting to prevent G4StateManager from deleting us
        CELER_LOG_LOCAL(debug)
            << "Deregistering state dependent " << static_cast<void*>(this)
            << " on " << local_stream_;
        manager_->DeregisterDependent(this);
    }

    if (mode_ == Mode::lifecycle)
    {
        this->notify_lifecycle(change);
    }
    else if (change == GeantStateChange::end_program)
    {
        // Preserve the legacy raw-callback behavior: existing diagnostic users
        // may destroy this object from the end-program callback.
        auto local_stream = local_stream_;
        auto cb = std::move(cb_);
        cb(local_stream, change);
    }
    else
    {
        this->cb_(local_stream_, change);
    }
    constexpr bool success{true};
    return success;
}

//---------------------------------------------------------------------------//
/*!
 * Dispatch filtered Celeritas lifecycle notifications.
 *
 * This suppresses Geant4 run-manager ordering details from automatic offload
 * lifecycle callbacks: the MT manager thread never emits end-run, workers do
 * not emit end-program, and duplicate begin/end run transitions on a local
 * state monitor are collapsed.
 */
void StateDependent::notify_lifecycle(GeantStateChange change)
{
    bool const is_mt = G4Threading::IsMultithreadedApplication();
    // A null stream is the MT manager thread; serial and worker callbacks have
    // a concrete local stream ID.
    bool const is_manager = !local_stream_;

    switch (change)
    {
        case GeantStateChange::begin_run:
            if (!active_run_)
            {
                cb_(local_stream_, change);
                active_run_ = true;
            }
            break;
        case GeantStateChange::end_run:
            // The MT manager owns shared state that must live until
            // end_program. Only serial/worker monitors emit end_run lifecycle
            // events, and only after they emitted begin_run.
            if (active_run_ && !is_manager)
            {
                cb_(local_stream_, change);
                active_run_ = false;
            }
            break;
        case GeantStateChange::end_program:
            // Serial uses one monitor for the whole lifecycle. In MT, worker
            // monitors finalize local state at end_run, so only the manager
            // emits end_program for shared cleanup.
            if (!is_mt || is_manager)
            {
                cb_(local_stream_, change);
            }
            active_run_ = false;
            break;
        default:
            break;
    }
}

//---------------------------------------------------------------------------//
/*!
 * Get a string corresponding to a Geant4 state change.
 */
char const* to_cstring(GeantStateChange value)
{
    static EnumStringMapper<GeantStateChange> const to_cstring_impl{
        "begin_program",
        "initialize",
        "begin_init",
        "internal_init",
        "end_init",
        "begin_run",
        "begin_event",
        "end_event",
        "end_run",
        "end_program",
        "unknown",
    };
    return to_cstring_impl(value);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
