//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/g4/StateDependent.cc
//---------------------------------------------------------------------------//
#include "StateDependent.hh"

#include <G4StateManager.hh>
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
StateDependent::StateDependent(LocalGeantStateChangeFunc cb)
    : local_stream_{geant_stream()}
    , cb_{std::move(cb)}
    , manager_{G4StateManager::GetStateManager()}
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

    this->cb_(local_stream_, change);
    constexpr bool success{true};
    return success;
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
