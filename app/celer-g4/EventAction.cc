//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/EventAction.cc
//---------------------------------------------------------------------------//
#include "EventAction.hh"

#include <type_traits>
#include <G4Event.hh>

#include "corecel/Macros.hh"

#include "GlobalSetup.hh"
#include "RootIO.hh"

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Construct with thread-local Celeritas data.
 */
EventAction::EventAction(SPConstParams params,
                         SPTransporter transport,
                         SPDiagnostics diagnostics)
    : params_(params)
    , transport_(transport)
    , diagnostics_{std::move(diagnostics)}
{
    CELER_EXPECT(params_);
    CELER_EXPECT(transport_);
    CELER_EXPECT(diagnostics_);
}

//---------------------------------------------------------------------------//
/*!
 * Inform Celeritas of the new event's ID.
 */
void EventAction::BeginOfEventAction(G4Event const* event)
{
    CELER_LOG_LOCAL(debug) << "Starting event " << event->GetEventID();

    get_event_time_ = {};

    if (SharedParams::CeleritasDisabled())
        return;

    // Set event ID in local transporter
    transport_->SetEventId(event->GetEventID());
}

//---------------------------------------------------------------------------//
/*!
 * Flush all offloaded tracks before ending the event.
 */
void EventAction::EndOfEventAction(G4Event const* event)
{
    CELER_EXPECT(event);

    if (!SharedParams::CeleritasDisabled())
    {
        // Transport any tracks left in the buffer
        transport_->Flush();
    }

    if (RootIO::use_root())
    {
        // Write sensitive hits
        RootIO::Instance()->Write(event);
    }

    // Record the time for this event
    diagnostics_->Timer()->RecordEventTime(get_event_time_());

    CELER_LOG_LOCAL(debug) << "Finished event " << event->GetEventID();
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
