//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/UserActionIntegration.cc
//---------------------------------------------------------------------------//
#include "UserActionIntegration.hh"

#include <G4Event.hh>
#include <G4Threading.hh>
#include <G4Track.hh>

#include "ExceptionConverter.hh"

#include "detail/IntegrationSingleton.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Access the singleton.
 */
UserActionIntegration& UserActionIntegration::Instance()
{
    static UserActionIntegration uai;
    return uai;
}

//---------------------------------------------------------------------------//
/*!
 * Edit options before starting the run.
 */
void UserActionIntegration::SetOptions(SetupOptions&& opts)
{
    detail::IntegrationSingleton::instance().setup_options(std::move(opts));
}

//---------------------------------------------------------------------------//
/*!
 * Initialize during ActionInitialization on non-worker thread in MT mode.
 */
void UserActionIntegration::BuildForMaster()
{
    CELER_VALIDATE(
        G4Threading::IsMasterThread()
            && G4Threading::IsMultithreadedApplication(),
        << R"(BuildForMaster called from a worker thread or non-MT code)");

    detail::IntegrationSingleton::instance().initialize_logger();
}

//---------------------------------------------------------------------------//
/*!
 * Initialize during ActionInitialization on worker thread or no-MT mode.
 */
void UserActionIntegration::Build()
{
    if (G4Threading::IsMasterThread())
    {
        CELER_VALIDATE(!G4Threading::IsMultithreadedApplication(),
                       << "cannot call Integration::Build from worker thread "
                          "in a multithreaded application");

        detail::IntegrationSingleton::instance().initialize_logger();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Start the run.
 */
void UserActionIntegration::BeginOfRunAction(G4Run const*)
{
    auto& singleton = detail::IntegrationSingleton::instance();

    if (G4Threading::IsMasterThread())
    {
        singleton.initialize_shared_params();
    }

    singleton.initialize_local_transporter();
}

//---------------------------------------------------------------------------//
/*!
 * Send Celeritas the event ID.
 */
void UserActionIntegration::BeginOfEventAction(G4Event const* event)
{
    auto& local = detail::IntegrationSingleton::local_transporter();
    if (!local)
        return;

    // Set event ID in local transporter and reseed RNG for reproducibility
    CELER_TRY_HANDLE(local.InitializeEvent(event->GetEventID()),
                     ExceptionConverter{"celer.event.begin"});
}

//---------------------------------------------------------------------------//
/*!
 * Send tracks to Celeritas if applicable and "StopAndKill" if so.
 */
void UserActionIntegration::PreUserTrackingAction(G4Track* track)
{
    CELER_EXPECT(track);

    auto& singleton = detail::IntegrationSingleton::instance();
    auto const mode = singleton.shared_params().mode();
    if (mode == SharedParams::Mode::disabled)
        return;

    auto const& particles = singleton.shared_params().OffloadParticles();
    if (std::find(particles.begin(), particles.end(), track->GetDefinition())
        != particles.end())
    {
        if (mode == SharedParams::Mode::enabled)
        {
            // Celeritas is transporting this track
            CELER_TRY_HANDLE(
                detail::IntegrationSingleton::local_transporter().Push(*track),
                ExceptionConverter("celer.track.push",
                                   &singleton.shared_params()));
        }

        // Either "pushed" or we're in kill_offload mode
        track->SetTrackStatus(fStopAndKill);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Flush offloaded tracks from Celeritas.
 */
void UserActionIntegration::EndOfEventAction(G4Event const*)
{
    auto& local = detail::IntegrationSingleton::local_transporter();
    if (!local)
        return;

    auto& singleton = detail::IntegrationSingleton::instance();
    CELER_TRY_HANDLE(
        local.Flush(),
        ExceptionConverter("celer.event.flush", &singleton.shared_params()));
}

//---------------------------------------------------------------------------//
/*!
 * End the run.
 */
void UserActionIntegration::EndOfRunAction(G4Run const*)
{
    CELER_LOG_LOCAL(status) << "Finalizing Celeritas";

    auto& singleton = detail::IntegrationSingleton::instance();

    // Remove local transporter
    singleton.finalize_local_transporter();

    if (G4Threading::IsMasterThread())
    {
        singleton.finalize_shared_params();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Only allow the singleton to construct.
 */
UserActionIntegration::UserActionIntegration() = default;

//---------------------------------------------------------------------------//
}  // namespace celeritas
