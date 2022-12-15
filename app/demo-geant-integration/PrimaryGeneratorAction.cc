//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-geant-integration/PrimaryGeneratorAction.cc
//---------------------------------------------------------------------------//
#include "PrimaryGeneratorAction.hh"

#include <G4ParticleTable.hh>
#include <G4SystemOfUnits.hh>
#include <G4ThreeVector.hh>

namespace demo_geant
{
//---------------------------------------------------------------------------//
/*!
 * Construct by setting up a default particle gun.
 */
PrimaryGeneratorAction::PrimaryGeneratorAction()
{
    auto g4particle_def = G4ParticleTable::GetParticleTable()->FindParticle(11);
    gun_.reset(new G4ParticleGun());
    gun_->SetParticleDefinition(g4particle_def);
    gun_->SetParticleEnergy(500 * MeV);
    gun_->SetParticlePosition(G4ThreeVector{0, 0, 0});          // origin
    gun_->SetParticleMomentumDirection(G4ThreeVector{1, 0, 0}); // +x
}

//---------------------------------------------------------------------------//
/*!
 * Construct by initializing the HepMC3 reader.
 */
PrimaryGeneratorAction::PrimaryGeneratorAction(
    std::shared_ptr<G4VPrimaryGenerator> hepmc3_reader)
    : hepmc3_reader_(hepmc3_reader)
{
}

//---------------------------------------------------------------------------//
/*!
 * Generate primaries from HepMC3 or particle gun.
 */
void PrimaryGeneratorAction::GeneratePrimaries(G4Event* event)
{
    if (gun_)
    {
        gun_->GeneratePrimaryVertex(event);
    }

    else
    {
        hepmc3_reader_->GeneratePrimaryVertex(event);
    }
}

//---------------------------------------------------------------------------//
} // namespace demo_geant
