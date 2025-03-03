//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/EventAction.cc
//---------------------------------------------------------------------------//
#include "EventAction.hh"

#include <type_traits>
#include <G4Event.hh>

#include "corecel/Macros.hh"
#include "accel/ExceptionConverter.hh"

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

    if (params_->mode() != SharedParams::Mode::enabled)
        return;

    // Set event ID in local transporter and reseed Celerits RNG
    ExceptionConverter call_g4exception{"celer.event.begin"};
    CELER_TRY_HANDLE(transport_->InitializeEvent(event->GetEventID()),
                     call_g4exception);
}

//---------------------------------------------------------------------------//
/*!
 * Flush all offloaded tracks before ending the event.
 */
void EventAction::EndOfEventAction(G4Event const* event)
{
    CELER_EXPECT(event);

    if (params_->mode() == SharedParams::Mode::enabled)
    {
        // Transport any tracks left in the buffer
        ExceptionConverter call_g4exception{"celer.event.flush", params_.get()};
        CELER_TRY_HANDLE(transport_->Flush(), call_g4exception);
    }

    if (GlobalSetup::Instance()->root_sd_io())
    {
        // Write sensitive hits
        RootIO::Instance()->Write(event);
    }

    // Record the time for this event
    diagnostics_->timer()->RecordEventTime(get_event_time_());

    CELER_LOG_LOCAL(debug) << "Finished event " << event->GetEventID();
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
