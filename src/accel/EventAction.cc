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
#include "corecel/io/Logger.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Initialize Celeritas.
 */
void EventAction::BeginOfEventAction(const G4Event* event)
{
    CELER_EXPECT(event);
    CELER_LOG_LOCAL(debug) << "EventAction::BeginOfEventAction for event "
                           << event->GetEventID();

    // Set event ID in local transporter
}

//---------------------------------------------------------------------------//
/*!
 * Finalize Celeritas.
 */
void EventAction::EndOfEventAction(const G4Event*)
{
    CELER_LOG_LOCAL(debug) << "EventAction::EndOfEventAction";

    // Transport any tracks left in the buffer
}

//---------------------------------------------------------------------------//
} // namespace celeritas
