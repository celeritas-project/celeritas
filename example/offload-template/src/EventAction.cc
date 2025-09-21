//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file offload-template/src/EventAction.cc
//---------------------------------------------------------------------------//
#include "EventAction.hh"

#include <G4Event.hh>
#include <G4SystemOfUnits.hh>
#include <accel/TrackingManagerIntegration.hh>
#include <corecel/Assert.hh>
#include <corecel/io/Logger.hh>
#include <corecel/sys/TraceCounter.hh>

#include "StepDiagnostic.hh"

namespace celeritas
{
namespace example
{
namespace
{
//---------------------------------------------------------------------------//
//! Safely provide global access to the step diagnostic
EventAction::SPStepDiagnostic& step_diagnostic()
{
    static EventAction::SPStepDiagnostic sd;
    return sd;
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * From MakeCelerOptions during setup on master, set the step diagnostic.
 *
 * This should only be called once from the main thread during BeginRun via
 * MakeCelerOptions.
 */
void EventAction::SetStepDiagnostic(SPStepDiagnostic&& diag)
{
    CELER_EXPECT(diag);
    CELER_LOG(debug) << "Setting step diagnostic";
    step_diagnostic() = std::move(diag);
}

//---------------------------------------------------------------------------//
/*!
 * During problem destruction, clear the diagnostic.
 *
 * This should only be called once from the main thread during EndRun.
 */
void EventAction::ClearStepDiagnostic()
{
    CELER_LOG(debug) << "Clearing step diagnostic";
    step_diagnostic() = nullptr;
}

//---------------------------------------------------------------------------//
/*!
 * At the end of each event, copy statistics from the local Celeritas state.
 */
void EventAction::BeginOfEventAction(G4Event const* event)
{
    trace_counter("event", event->GetEventID());

    // Start the range for the event
    profile_this_.emplace("Event");
}

//---------------------------------------------------------------------------//
/*!
 * At the end of each event, copy statistics from the local Celeritas state.
 */
void EventAction::EndOfEventAction(G4Event const* event)
{
    CELER_VALIDATE(step_diagnostic(), << "step diagnostic was not constructed");

    // End the range for the event
    profile_this_.reset();

    // Note that the diagnostic is *const* (unmodified, thread safe) and the
    // state data (thread local!) is mutable
    auto const& diagnostic = *step_diagnostic();
    auto& state = celeritas::TrackingManagerIntegration::Instance().GetState();

    // Get accumulated stats and prepare the state for the next event
    auto stats = diagnostic.GetAndReset(state);

    // clang-format off
    CELER_LOG_LOCAL(info) << "In event " << event->GetEventID() << ":\n"
        "  average step length = " << (stats.step_length / mm) / stats.num_steps << " mm\n"
        "  energy deposition = " << stats.energy_deposition / MeV << " MeV\n"
        "  with " << stats.num_primaries << " primaries\n"
        "  and " << stats.num_secondaries << " secondaries";
    // clang-format on
}

//---------------------------------------------------------------------------//
}  // namespace example
}  // namespace celeritas
