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
#include <G4MuonMinus.hh>
#include <G4MuonPlus.hh>
#include <G4Positron.hh>

#include "corecel/io/Logger.hh"

#include "SharedParams.hh"
#include "TrackingManager.hh"
#include "TrackingManagerIntegration.hh"

#include "detail/IntegrationSingleton.hh"

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
        G4MuonMinus::Definition(),
        G4MuonPlus::Definition(),
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
 * Construct from tracking manager integration.
 *
 * Since there's only ever one tracking manager integration, we can just use
 * the behind-the-hood objects.
 */
TrackingManagerConstructor::TrackingManagerConstructor(
    TrackingManagerIntegration* tmi)
    : TrackingManagerConstructor(
          &detail::IntegrationSingleton::instance().shared_params(), [](int) {
              return &detail::IntegrationSingleton::instance()
                          .local_transporter();
          })
{
    CELER_EXPECT(tmi == &TrackingManagerIntegration::Instance());
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

    CELER_LOG(debug) << "Activating tracking manager";

    // Note that error checking occurs here to provide better error messages
    CELER_VALIDATE(
        shared_ && get_local_,
        << R"(invalid null inputs given to TrackingManagerConstructor)");

    auto* transporter = this->get_local_transporter();
    CELER_VALIDATE(transporter, << "invalid null local transporter");

    // Create *thread-local* tracking manager with pointers to *global*
    // shared params and *thread-local* transporter.
    auto manager = std::make_unique<TrackingManager>(shared_, transporter);
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
/*!
 * Override default list of particles defined by \c ::OffloadParticles() .
 */
void TrackingManagerConstructor::SetOffloadParticles(
    Span<G4ParticleDefinition* const> subset)
{
    CELER_EXPECT(!subset.empty());
    auto const full_set = OffloadParticles();
    CELER_EXPECT(subset.size() <= full_set.size());

    auto is_valid = [&](G4ParticleDefinition const* particle) -> bool {
        for (auto valid_part : full_set)
        {
            CELER_EXPECT(valid_part);
            if (particle->GetPDGEncoding() == valid_part->GetPDGEncoding())
            {
                return true;
            }
        }
        return false;
    };

    // Ensure that every particle in the subset is available in Celeritas
    for (auto const* p : subset)
    {
        CELER_EXPECT(p);
        CELER_VALIDATE(is_valid(p),
                       << "Particle " << p->GetParticleName()
                       << " is not available in Celeritas");
    }

    // Override list
    offload_particles_ = subset;
}

//---------------------------------------------------------------------------//
/*!
 * Get the local transporter associated with the current thread ID.
 */
LocalTransporter* TrackingManagerConstructor::get_local_transporter() const
{
    CELER_EXPECT(get_local_);
    return this->get_local_(G4Threading::G4GetThreadId());
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
