//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file example/offload-template/src/EventAction.cc
//---------------------------------------------------------------------------//
#include "EventAction.hh"

#include "Celeritas.hh"
#include "G4Event.hh"

//---------------------------------------------------------------------------//
/*!
 * Construct empty.
 */
EventAction::EventAction() : G4UserEventAction() {}

//---------------------------------------------------------------------------//
/*!
 * Initialize event in Celeritas.
 */
void EventAction::BeginOfEventAction(G4Event const* event)
{
    CelerSimpleOffload().BeginOfEventAction(event);
}

//---------------------------------------------------------------------------//
/*!
 * Flush any remaining particles to Celeritas.
 */
void EventAction::EndOfEventAction(G4Event const* event)
{
    CelerSimpleOffload().EndOfEventAction(event);
}
