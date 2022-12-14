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
EventAction::EventAction() {}

//---------------------------------------------------------------------------//
/*!
 * Initialize Celeritas.
 */
void EventAction::BeginOfEventAction(const G4Event* event)
{
    CELER_EXPECT(event);

    // TODO: Set event ID in local transporter
}

//---------------------------------------------------------------------------//
/*!
 * Finalize Celeritas.
 */
void EventAction::EndOfEventAction(const G4Event*)
{
    // TODO: Transport any tracks left in the buffer
}

//---------------------------------------------------------------------------//
} // namespace celeritas
