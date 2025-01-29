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

#include "detail/StaticIntegrationData.hh"

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
    CELER_VALIDATE(
        !detail::shared_params(),
        << R"(options cannot be modified after Celeritas is constructed)");

    return detail::setup_options();
}

//---------------------------------------------------------------------------//
/*!
 * Initialize during ActionInitialization.
 *
 * \todo The query for CeleritasDisabled initializes the environment before
 * we've had a chance to load the user setup options. Make sure we can update
 * the environment *first* when refactoring the setup.
 *
 * \note In Geant4 threading, \em only MT mode on non-master thread has
 *   \c G4Threading::IsWorkerThread()==true. For MT mode, the master thread
 *   does not track any particles. For single-thread mode, the master thread
 *   \em does do work.
 */
void TrackingManagerIntegration::Build()
{
    auto* run_man = G4RunManager::GetRunManager();
    CELER_VALIDATE(
        run_man, << R"(Geant4 run manager was not initialized before build)");

    if (G4Threading::IsMasterThread())
    {
        CELER_VALIDATE(
            !shared_params(),
            << R"(build cannot be called from master thread more than once)");

        // Initialize multithread logger if run manager exists
        celeritas::self_logger() = celeritas::MakeMTLogger(*run_man);
        thread_managers.reserve(run_man->GetNumberOfThreads());
    }

    if (SharedParams::CeleritasDisabled())
    {
        CELER_LOG_LOCAL(debug)
            << R"(Not building tracking manager: Celeritas is disabled)";
        return;
    }

    if (!G4Threading::IsMultithreadedApplication()
        || !G4Threading::IsMasterThread())
    {
        // Set tracking manager on workers
        CELER_LOG_LOCAL(debug) << "Setting tracking manager";

        // Create *thread-local* tracking manager with pointers to *global*
        // shared params and *thread-local* transporter
        auto manager = std::make_unique<TrackingManager>(
            detail::shared_params(), detail::local_transporter());

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
}  // namespace celeritas
