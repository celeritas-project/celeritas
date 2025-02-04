//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/UserActionIntegration.cc
//---------------------------------------------------------------------------//
#include "UserActionIntegration.hh"

#include <G4Threading.hh>

#include "detail/IntegrationSingleton.hh"

namespace celeritas
{
#if 0
//---------------------------------------------------------------------------//
/*!
 * Access the singleton.
 */
UserActionIntegration& UserActionIntegration::instance()
{
    static UserActionIntegration uai;
    return uai;
}

//---------------------------------------------------------------------------//
/*!
 * Edit options before starting the run.
 */
SetupOptions& UserActionIntegration::Options()
{
    return detail::IntegrationSingleton::instance().setup_options();
}

//---------------------------------------------------------------------------//
/*!
 * Initialize during ActionInitialization on non-worker thread in MT mode.
 */
void UserActionIntegration::BuildForMaster()
{
    auto& singleton = detail::IntegrationSingleton::instance();
    singleton.initialize_shared_params();
}

//---------------------------------------------------------------------------//
/*!
 * Initialize during ActionInitialization on worker thread or no-MT mode.
 */
void UserActionIntegration::Build()
{
    auto& singleton = detail::IntegrationSingleton::instance();

    if (G4Threading::IsMasterThread())
    {
        CELER_VALIDATE(!G4Threading::IsMultithreadedApplication(),
                       << "cannot call Integration::Build from worker thread "
                          "in an MT application");
        singleton.initialize_shared_params();
    }

    singleton.initialize_local_transporter();
}

//---------------------------------------------------------------------------//
/*!
 * Start the run.
 */
void UserActionIntegration::BeginOfRunAction(G4Run const* run)
{
    // SimpleOffload does construction at beginning-of-run, not Build
}

//---------------------------------------------------------------------------//
/*!
 * Send Celeritas the event ID.
 */
void UserActionIntegration::BeginOfEventAction(G4Event const* event)
{
    // Set event ID in local transporter and reseed RNG for reproducibility
    ExceptionConverter call_g4exception{"celer.event.begin"};
    CELER_TRY_HANDLE(local_->InitializeEvent(event->GetEventID()),
                     call_g4exception);
}

//---------------------------------------------------------------------------//
/*!
 * Send tracks to Celeritas if applicable and "StopAndKill" if so.
 */
void UserActionIntegration::PreUserTrackingAction(G4Track* track) {}

//---------------------------------------------------------------------------//
/*!
 * Flush offloaded tracks from Celeritas.
 */
void UserActionIntegration::EndOfEventAction(G4Event const* event) {}

//---------------------------------------------------------------------------//
/*!
 * End the run.
 */
void UserActionIntegration::EndOfRunAction(G4Run const* run) {}

//---------------------------------------------------------------------------//
/*!
 * Only allow the singleton to construct.
 */
UserActionIntegration::UserActionIntegration() = default;

//---------------------------------------------------------------------------//
#endif
}  // namespace celeritas
