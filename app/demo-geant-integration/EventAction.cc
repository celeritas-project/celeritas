//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-geant-integration/EventAction.cc
//---------------------------------------------------------------------------//
#include "EventAction.hh"

#include <G4Event.hh>

namespace demo_geant
{
//---------------------------------------------------------------------------//
/*!
 * Construct with Celeritas shared data.
 */
EventAction::EventAction(SPTransporter transport) : transport_(transport) {}

//---------------------------------------------------------------------------//
/*!
 * Initialize Celeritas.
 */
void EventAction::BeginOfEventAction(const G4Event* event)
{
    // Set event ID in local transporter
    transport_->SetEventId(event->GetEventID());
}

//---------------------------------------------------------------------------//
/*!
 * Finalize Celeritas.
 */
void EventAction::EndOfEventAction(const G4Event*)
{
    // Transport any tracks left in the buffer
    transport_->Flush();
}

//---------------------------------------------------------------------------//
} // namespace demo_geant
