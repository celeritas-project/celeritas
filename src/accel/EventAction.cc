//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/EventAction.cc
//---------------------------------------------------------------------------//
#include "EventAction.hh"

#include <G4Event.hh>

#include "corecel/Assert.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with Celeritas shared data.
 */
EventAction::EventAction(SPParams params, SPTransporter transport)
    : params_(params), transport_(transport)
{
    CELER_EXPECT(params_);
    CELER_EXPECT(transport_);
}

//---------------------------------------------------------------------------//
/*!
 * Initialize Celeritas.
 */
void EventAction::BeginOfEventAction(const G4Event* event)
{
    CELER_EXPECT(event);

    // Set event ID in local transporter
    transport_->set_event(EventId{EventId::size_type(event->GetEventID())});
}

//---------------------------------------------------------------------------//
/*!
 * Finalize Celeritas.
 */
void EventAction::EndOfEventAction(const G4Event*)
{
    // Transport any tracks left in the buffer
    transport_->flush();
}

//---------------------------------------------------------------------------//
} // namespace celeritas
