//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/TrackingManagerIntegration.cc
//---------------------------------------------------------------------------//
#include "TrackingManagerIntegration.hh"

#include <G4ParticleDefinition.hh>
#include <G4Run.hh>
#include <G4Threading.hh>

#include "geocel/GeantUtils.hh"

#include "detail/IntegrationSingleton.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Access the singleton.
 */
TrackingManagerIntegration& TrackingManagerIntegration::instance()
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
void TrackingManagerIntegration::BuildForMaster() {}

//---------------------------------------------------------------------------//
/*!
 * Initialize during ActionInitialization.
 */
void TrackingManagerIntegration::Build()
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

    if (singleton.shared_params()
        && (!G4Threading::IsMultithreadedApplication()
            || G4Threading::IsWorkerThread()))
    {
        // Set tracking manager on workers when Celeritas is enabled
        CELER_LOG_LOCAL(debug) << "Setting tracking manager";

        // Create *thread-local* tracking manager with pointers to *global*
        // shared params and *thread-local* transporter
        auto manager = std::make_unique<TrackingManager>(
            singleton.shared_params(), singleton.local_transporter());

        for (G4ParticleDefinition* particle :
             detail::shared_params().offload_particles())
        {
            particle->SetTrackingManager(manager.get());
        }

        // Save thread manager so we can access the thread-local data later if
        // needed for testing/verification/etc
        auto thread_id = get_geant_thread_id();
        CELER_ASSERT(thread_id >= 0
                     && static_cast<std::size_t>(thread_id)
                            < thread_managers.size());
        thread_managers[thread_id] = std::move(manager);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Start the run.
 */
void TrackingManagerIntegration::BeginOfRunAction(G4Run const* run)
{
    // TODO: initialize shared params?
}

//---------------------------------------------------------------------------//
/*!
 * End the run.
 */
void TrackingManagerIntegration::EndOfRunAction(G4Run const* run)
{
    // TODO: finalize Celeritas
}

//---------------------------------------------------------------------------//
/*!
 * Only allow the singleton to construct.
 */
TrackingManagerIntegration::TrackingManagerIntegration() = default;

//---------------------------------------------------------------------------//
}  // namespace celeritas
