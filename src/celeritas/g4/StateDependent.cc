//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/g4/StateDependent.cc
//---------------------------------------------------------------------------//
#include "StateDependent.hh"

#include <G4StateManager.hh>

#include "corecel/Assert.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with a stream ID and state-change callback.
 */
StateDependent::StateDependent(StreamId sid, LocalStateChangeFunc cb)
    : local_stream_{sid}, cb_{std::move(cb)}
{
    CELER_EXPECT(cb_);
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
    StateDependent::GeantStateChange change
        = StateDependent::GeantStateChange::unknown;

    if ((prev == G4State_PreInit && state == G4State_Init)
        || (prev == G4State_Idle && state == G4State_Init))
    {
        change = StateDependent::GeantStateChange::initialize;
    }
    else if (prev == G4State_Idle && state == G4State_GeomClosed)
    {
        change = StateDependent::GeantStateChange::begin_run;
    }
    else if (prev == G4State_GeomClosed && state == G4State_EventProc)
    {
        change = StateDependent::GeantStateChange::begin_event;
    }
    else if (prev == G4State_EventProc && state == G4State_GeomClosed)
    {
        change = StateDependent::GeantStateChange::end_event;
    }
    else if (prev == G4State_GeomClosed && state == G4State_Idle)
    {
        change = StateDependent::GeantStateChange::end_run;
    }

    this->cb_(local_stream_, change);
    return true;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
