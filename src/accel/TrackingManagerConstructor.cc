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
#include <G4ParticleTable.hh>
#include <G4Positron.hh>

#include "corecel/io/Logger.hh"

#include "SharedParams.hh"
#include "TrackingManager.hh"
#include "TrackingManagerIntegration.hh"

#include "detail/IntegrationSingleton.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Return a subset of particles to be offload based on user-input.
 */
TrackingManagerConstructor::VecG4PD
OffloadParticleSubset(SetupOptions::VecPDG const& input_list)
{
    CELER_EXPECT(!input_list.empty());
    CELER_LOG(status) << "Loading user-defined set of particles to offload";

    auto const full_set = TrackingManagerConstructor::OffloadParticles();
    CELER_VALIDATE(input_list.size() <= full_set.size(),
                   << "List of particles to be offloaded defined in "
                      "SetupOptions is larger than the number of available "
                      "particles in Celeritas");

    auto find = [&full_set](int pdg) -> G4ParticleDefinition* {
        auto it = std::find_if(
            full_set.begin(), full_set.end(), [&pdg](G4ParticleDefinition* p) {
                return (p->GetPDGEncoding() == pdg);
            });
        return *it;
    };

    // Create vector of particles from user-defined list of PDGs
    std::vector<G4ParticleDefinition*> result;
    for (auto const pdg : input_list)
    {
        CELER_VALIDATE(pdg != 0, << "PDG must not be zero");
        auto* p = find(pdg);
        CELER_VALIDATE(p,
                       << "Particle with PDG = " << pdg
                       << " is not available in Celeritas");
        CELER_LOG(info) << "Loaded particle " << p->GetParticleName();
        result.push_back(p);
    }
    return result;
}
//---------------------------------------------------------------------------//
};  // namespace

//---------------------------------------------------------------------------//
/*!
 * Get a list of supported particles that will be offloaded.
 */
TrackingManagerConstructor::VecG4PD
TrackingManagerConstructor::OffloadParticles()
{
    static G4ParticleDefinition* const supported_particles[] = {
        G4Electron::Definition(),
        G4Positron::Definition(),
        G4Gamma::Definition(),
        G4MuonMinus::Definition(),
        G4MuonPlus::Definition(),
    };

    return {std::begin(supported_particles), std::end(supported_particles)};
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
 * Construct particles before \c ::ConstructProcess since any
 * \c G4ParticleDefinition other than ions and shortlived must be created
 * during Geant4's \c Pre_Init state.
 */
void TrackingManagerConstructor::ConstructParticle()
{
    auto& singleton = detail::IntegrationSingleton::instance();
    auto const& user_offload = singleton.setup_options().offload_particles;
    // Construction of particles happens at offload_particles_ assignment
    offload_particles_ = user_offload.empty()
                             ? OffloadParticles()
                             : OffloadParticleSubset(user_offload);
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

    // Load list of particles to offload based on user-options
    for (auto* p : offload_particles_)
    {
        CELER_EXPECT(p);
        // Memory for the tracking manager should be freed in
        // G4VUserPhysicsList::TerminateWorker from G4WorkerRunManager
        // by constructing a 'set' of all tracking managers.
        // (Note that it is leaked in Geant4 11.0 and 11.1 for MT mode.)
        p->SetTrackingManager(manager ? manager.release() : manager_ptr);
        CELER_LOG(info) << "Added particle \"" << p->GetParticleName()
                        << "\" to Tracking Manager ";
    }
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
