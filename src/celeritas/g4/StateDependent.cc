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
 */
StateDependent::StateDependent(LocalStateChangeFunc cb)
    : local_stream_{geant_stream()}, cb_{std::move(cb)}
{
    CELER_EXPECT(cb_);
    CELER_LOG_LOCAL(error) << "Creating state dependent " << (void*)this;
}

//---------------------------------------------------------------------------//
/*!
 * Log on deletion
 */
StateDependent::~StateDependent()
{
    CELER_LOG_LOCAL(error) << "Deleting state dependent " << (void*)this;
}

//---------------------------------------------------------------------------//
/*!
 * Dispatch a state transition notification to the user callback.
 */
G4bool StateDependent::Notify(G4ApplicationState state)
{
    G4StateManager* sm = G4StateManager::GetStateManager();
    CELER_ASSERT(sm);
    G4ApplicationState prev = sm->GetPreviousState();
    // Map (previous, requested) Geant4 states to our semantic enum.
    auto change = GeantStateChange::unknown;

    switch (state)
    {
        case G4State_PreInit:
            // Constructing run kernel
            break;
        case G4State_Init:
            // Initializing
            if (prev == G4State_PreInit || prev == G4State_Idle)
            {
                change = GeantStateChange::initialize;
            }
            break;
        case G4State_Idle:
            // Completing a run
            if (prev == G4State_GeomClosed)
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
    };
    if (change == GeantStateChange::unknown)
    {
        CELER_LOG_LOCAL(warning)
            << "Unknown state change: " << sm->GetStateString(prev) << "->"
            << sm->GetStateString(state);
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
