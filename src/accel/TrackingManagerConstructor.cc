//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/TrackingManagerConstructor.cc
//---------------------------------------------------------------------------//
#include "TrackingManagerConstructor.hh"

#include <G4BuilderType.hh>
#include <G4Version.hh>
#if G4VERSION_NUMBER >= 1100
#    include "TrackingManager.hh"
#endif

#include "corecel/io/Join.hh"
#include "corecel/io/Logger.hh"

#include "SharedParams.hh"
#include "TrackingManagerIntegration.hh"

#include "detail/IntegrationSingleton.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct name and mode.
 *
 * Error checking is deferred until ConstructProcess.
 */
TrackingManagerConstructor::TrackingManagerConstructor(
    SharedParams const* shared, LocalTransporterFromThread get_local)
    : G4VPhysicsConstructor("offload-physics")
    , shared_(shared)
    , get_local_(get_local)
{
    // The special "unknown" type will not conflict with any other physics
    this->SetPhysicsType(G4BuilderType::bUnknown);

    CELER_VALIDATE(G4VERSION_NUMBER >= 1100,
                   << "the current version of Geant4 (" << G4VERSION_NUMBER
                   << ") is too old to support the tracking manager offload "
                      "interface (11.0 or higher is required)");
}

//---------------------------------------------------------------------------//
/*!
 * Construct from tracking manager integration.
 *
 * Since there's only ever one tracking manager integration, we can just use
 * the behind-the-hood objects.
 *
 * \note When calling from a serial run manager in a threaded G4 build, the
 * thread ID is \c G4Threading::MASTER_ID (-1). When calling from the run
 * manager of a non-threaded G4 build, the thread is \c
 * G4Threading::SEQUENTIAL_ID (-2).
 */
TrackingManagerConstructor::TrackingManagerConstructor(
    TrackingManagerIntegration* tmi)
    : TrackingManagerConstructor(
          &detail::IntegrationSingleton::instance().shared_params(),
          [](int tid) {
              CELER_EXPECT(tid >= 0
                           || !G4Threading::IsMultithreadedApplication());
              return &detail::IntegrationSingleton::instance()
                          .local_transporter();
          })
{
    CELER_EXPECT(tmi == &TrackingManagerIntegration::Instance());
}

//---------------------------------------------------------------------------//
/*!
 * Construct particles and determine which to offload.
 *
 * This is called \em early in the application, when the physics list is passed
 * to the run manager. It is only called once on a multithreaded run,
 * during Geant4's \c Pre_Init state.
 */
void TrackingManagerConstructor::ConstructParticle()
{
    // Construction of particles happens at offload_particles_ assignment,
    // since it will instantiate the G4Particle::Definition() singletons
    auto& is = detail::IntegrationSingleton::instance();
    auto& opts = is.setup_options();
    offload_particles_ = opts.offload_particles
                             ? is.offloaded_particles()
                             : SharedParams::default_offload_particles();
}

//---------------------------------------------------------------------------//
/*!
 * Build and attach tracking manager.
 */
void TrackingManagerConstructor::ConstructProcess()
{
    if (SharedParams::GetMode() == SharedParams::Mode::disabled)
    {
        CELER_LOG(debug)
            << R"(Skipping tracking manager since Celeritas is disabled)";
        return;
    }

    CELER_LOG_LOCAL(debug) << "Activating tracking manager";

    // Note that error checking occurs here to provide better error messages
    CELER_VALIDATE(
        shared_ && get_local_,
        << R"(invalid null inputs given to TrackingManagerConstructor)");

    LocalTransporter* transporter{nullptr};

    if (G4Threading::IsWorkerThread()
        || !G4Threading::IsMultithreadedApplication())
    {
        // Don't create or access local transporter on master thread
        transporter = this->get_local_(G4Threading::G4GetThreadId());
        CELER_VALIDATE(transporter, << "invalid null local transporter");
    }

#if G4VERSION_NUMBER >= 1100
    // Create *thread-local* tracking manager with pointers to *global*
    // shared params and *thread-local* transporter.
    auto manager = std::make_unique<TrackingManager>(shared_, transporter);
    auto* manager_ptr = manager.get();

    for (auto* p : offload_particles_)
    {
        // Memory for the tracking manager should be freed in
        // G4VUserPhysicsList::TerminateWorker from G4WorkerRunManager
        // by constructing a 'set' of all tracking managers.
        // (Note that it is leaked in Geant4 11.0 and 11.1 for MT mode.)
        p->SetTrackingManager(manager ? manager.release() : manager_ptr);
    }
    CELER_LOG(info) << "Built Celeritas tracking managers for "
                    << join(offload_particles_.begin(),
                            offload_particles_.end(),
                            ", ",
                            [](G4ParticleDefinition const* pd) {
                                return pd->GetParticleName();
                            });
#else
    // Constructor should've prevented this
    CELER_ASSERT_UNREACHABLE();
#endif
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
