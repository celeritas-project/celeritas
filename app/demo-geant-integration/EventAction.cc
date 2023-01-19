//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
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

#include "GlobalSetup.hh"
#include "HitRootIO.hh"

namespace demo_geant
{
//---------------------------------------------------------------------------//
/*!
 * Construct with thread-local Celeritas data.
 */
EventAction::EventAction(SPTransporter transport) : transport_(transport) {}

//---------------------------------------------------------------------------//
/*!
 * Inform Celeritas of the new event's ID.
 */
void EventAction::BeginOfEventAction(G4Event const* event)
{
    // Set event ID in local transporter
    celeritas::ExceptionConverter call_g4exception{"celer0002"};
    CELER_TRY_HANDLE(transport_->SetEventId(event->GetEventID()),
                     call_g4exception);
}

//---------------------------------------------------------------------------//
/*!
 * Flush all offloaded tracks before ending the event.
 */
void EventAction::EndOfEventAction(G4Event const* event)
{
    // Transport any tracks left in the buffer
    celeritas::ExceptionConverter call_g4exception{"celer0004"};
    CELER_TRY_HANDLE(transport_->Flush(), call_g4exception);

    if (GlobalSetup::Instance()->GetSetupOptions()->sd.write_hits)
    {
        // Write sensitive hits
        HitRootIO::GetInstance()->WriteHits(event);
    }
}

//---------------------------------------------------------------------------//
}  // namespace demo_geant
