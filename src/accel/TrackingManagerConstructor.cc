//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/TrackingManagerConstructor.cc
//---------------------------------------------------------------------------//
#include "TrackingManagerConstructor.hh"

#include <G4BuilderType.hh>
#include <G4Electron.hh>
#include <G4Gamma.hh>
#include <G4Positron.hh>

#include "corecel/io/Logger.hh"

#include "SharedParams.hh"
#include "TrackingManager.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Get a list of supported particles that will be offloaded.
 */
Span<G4ParticleDefinition* const> TrackingManagerConstructor::OffloadParticles()
{
    static G4ParticleDefinition* const supported_particles[] = {
        G4Electron::Definition(),
        G4Positron::Definition(),
        G4Gamma::Definition(),
    };

    return make_span(supported_particles);
}

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
}

//---------------------------------------------------------------------------//
/*!
 * Build and attach tracking manager.
 */
void TrackingManagerConstructor::ConstructProcess()
{
    if (SharedParams::GetMode() == SharedParams::Mode::disabled)
    {
        CELER_LOG_LOCAL(debug)
            << R"(Skipping tracking manager since Celeritas is disabled)";
        return;
    }

    CELER_LOG_LOCAL(debug) << "Activating tracking manager";

    CELER_VALIDATE(shared_ && get_local_,
                   << "invalid null inputs given to "
                      "TrackingManagerConstructor");
    auto* local_transporter = get_local_(G4Threading::G4GetThreadId());
    CELER_VALIDATE(local_transporter, << "invalid null local transporter");

    // Create *thread-local* tracking manager with pointers to *global*
    // shared params and *thread-local* transporter.
    auto manager
        = std::make_unique<TrackingManager>(shared_, local_transporter);
    auto* manager_ptr = manager.get();

    for (auto* p : OffloadParticles())
    {
        CELER_EXPECT(p);
        // Memory for the tracking manager should be freed in
        // G4VUserPhysicsList::TerminateWorker from G4WorkerRunManager
        // by constructing a 'set' of all tracking managers.
        // (Note that it is leaked in Geant4 11.0 and 11.1 for MT mode.)
        p->SetTrackingManager(manager ? manager.release() : manager_ptr);
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
