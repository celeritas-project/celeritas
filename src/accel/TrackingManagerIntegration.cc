//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/TrackingManagerIntegration.cc
//---------------------------------------------------------------------------//
#include "TrackingManagerIntegration.hh"

#include <memory>
#include <G4ParticleDefinition.hh>
#include <G4Run.hh>
#include <G4Threading.hh>

#include "geocel/GeantUtils.hh"

#include "TrackingManager.hh"

#include "detail/IntegrationSingleton.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Access the public-facing integration singleton.
 */
TrackingManagerIntegration& TrackingManagerIntegration::Instance()
{
    static TrackingManagerIntegration tmi;
    return tmi;
}

//---------------------------------------------------------------------------//
/*!
 * Edit options before starting the run.
 */
SetupOptions& TrackingManagerIntegration::Options()
{
    return detail::IntegrationSingleton::instance().setup_options();
}

//---------------------------------------------------------------------------//
/*!
 * Initialize during ActionInitialization on non-worker thread in MT mode.
 */
void TrackingManagerIntegration::BuildForMaster()
{
    CELER_VALIDATE(
        G4Threading::IsMasterThread()
            || G4Threading::IsMultithreadedApplication(),
        << R"(BuildForMaster called from a worker thread or non-MT code)");

    detail::IntegrationSingleton::instance().initialize_logger();
}

//---------------------------------------------------------------------------//
/*!
 * Initialize during ActionInitialization.
 */
void TrackingManagerIntegration::Build()
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
void TrackingManagerIntegration::BeginOfRunAction(G4Run const*)
{
    auto& singleton = detail::IntegrationSingleton::instance();

    if (G4Threading::IsMasterThread())
    {
        singleton.initialize_shared_params();
    }

    bool enable_offload = singleton.initialize_local_transporter();

    if (enable_offload)
    {
        // Set tracking manager on workers when Celeritas is not fully disabled
        CELER_LOG_LOCAL(debug) << "Setting tracking manager";

        // Create *thread-local* tracking manager with pointers to *global*
        // shared params and *thread-local* transporter.
        // Memory for the tracking manager should be freed in
        // G4VUserPhysicsList::TerminateWorker from
        // G4WorkerRunManager::~G4WorkerRunManager (note that it is leaked in
        // Geant4 11.0 and 11.1)
        auto manager = std::make_unique<TrackingManager>(
            &singleton.shared_params(), &singleton.local_transporter());
        auto* manager_ptr = manager.get();

        for (G4ParticleDefinition* particle :
             singleton.shared_params().OffloadParticles())
        {
            particle->SetTrackingManager(manager ? manager.release()
                                                 : manager_ptr);
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * End the run.
 */
void TrackingManagerIntegration::EndOfRunAction(G4Run const*)
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
TrackingManagerIntegration::TrackingManagerIntegration() = default;

//---------------------------------------------------------------------------//
}  // namespace celeritas
