//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-geant-integration/EventAction.cc
//---------------------------------------------------------------------------//
#include "EventAction.hh"

#include <type_traits>
#include <G4Event.hh>

#include "corecel/Macros.hh"
#include "accel/ExceptionConverter.hh"

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
    celeritas::ExceptionConverter call_g4exception{"celer0002"};
    CELER_TRY_HANDLE(transport_->SetEventId(event->GetEventID()),
                     call_g4exception);
}

//---------------------------------------------------------------------------//
/*!
 * Finalize Celeritas.
 */
void EventAction::EndOfEventAction(const G4Event*)
{
    // Transport any tracks left in the buffer
    celeritas::ExceptionConverter call_g4exception{"celer0004"};
    CELER_TRY_HANDLE(transport_->Flush(), call_g4exception);
}

//---------------------------------------------------------------------------//
} // namespace demo_geant
