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
#include "corecel/sys/Environment.hh"
#include "accel/ExceptionConverter.hh"

#include "GlobalSetup.hh"
#include "HitRootIO.hh"

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Construct with thread-local Celeritas data.
 */
EventAction::EventAction(SPConstParams params, SPTransporter transport)
    : params_(params)
    , transport_(transport)
    , disable_offloading_(!celeritas::getenv("CELER_DISABLE").empty())
{
    CELER_EXPECT(params_);
    CELER_EXPECT(transport_);
}

//---------------------------------------------------------------------------//
/*!
 * Inform Celeritas of the new event's ID.
 */
void EventAction::BeginOfEventAction(G4Event const* event)
{
    CELER_LOG_LOCAL(debug) << "Starting event " << event->GetEventID();

    if (disable_offloading_)
        return;

    // Set event ID in local transporter
    ExceptionConverter call_g4exception{"celer0002"};
    CELER_TRY_HANDLE(transport_->SetEventId(event->GetEventID()),
                     call_g4exception);
}

//---------------------------------------------------------------------------//
/*!
 * Flush all offloaded tracks before ending the event.
 */
void EventAction::EndOfEventAction(G4Event const* event)
{
    CELER_EXPECT(event);

    if (!disable_offloading_)
    {
        // Transport any tracks left in the buffer
        ExceptionConverter call_g4exception{"celer0004", params_.get()};
        CELER_TRY_HANDLE(transport_->Flush(), call_g4exception);
    }

    if (GlobalSetup::Instance()->GetWriteSDHits())
    {
        // Write sensitive hits
        HitRootIO::Instance()->WriteHits(event);
    }

    CELER_LOG_LOCAL(debug) << "Finished event " << event->GetEventID();
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
