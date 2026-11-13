//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/g4/SupportedDecayPhysics.cc
//---------------------------------------------------------------------------//
#include "SupportedDecayPhysics.hh"

#include <G4BuilderType.hh>
#include <G4Decay.hh>
#include <G4LeptonConstructor.hh>
#include <G4MuonMinus.hh>
#include <G4MuonPlus.hh>
#include <G4ParticleDefinition.hh>
#include <G4PhysicsListHelper.hh>

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with physics options.
 */
SupportedDecayPhysics::SupportedDecayPhysics(Options const& options)
    : G4VPhysicsConstructor("decay", bDecay)
{
    CELER_VALIDATE(options.muon && options.muon->decay,
                   << "cannot construct SupportedDecayPhysics when muon decay "
                      "is disabled");
}

//---------------------------------------------------------------------------//
/*!
 * Set up muons and the leptons they decay to.
 */
void SupportedDecayPhysics::ConstructParticle()
{
    G4LeptonConstructor::ConstructParticle();
}

//---------------------------------------------------------------------------//
/*!
 * Add decay to supported particles.
 *
 * The muon decay table (\c G4MuonDecayChannel) is set by the particle
 * definition.
 */
void SupportedDecayPhysics::ConstructProcess()
{
    auto* ph = G4PhysicsListHelper::GetPhysicsListHelper();

    // Geant4 takes ownership of the process (shared by mu- and mu+) when it is
    // registered
    auto* decay = new G4Decay();
    for (G4ParticleDefinition* p :
         {static_cast<G4ParticleDefinition*>(G4MuonMinus::Definition()),
          static_cast<G4ParticleDefinition*>(G4MuonPlus::Definition())})
    {
        ph->RegisterProcess(decay, p);
    }
    CELER_LOG(debug) << "Using muon decay with G4Decay";
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
