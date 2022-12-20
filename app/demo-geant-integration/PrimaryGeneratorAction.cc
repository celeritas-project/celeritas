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
 * Set up particle gun
 */
PrimaryGeneratorAction::PrimaryGeneratorAction()
{
    auto g4particle_def = G4ParticleTable::GetParticleTable()->FindParticle(11);
    gun_.SetParticleDefinition(g4particle_def);
    gun_.SetParticleEnergy(500 * MeV);
    gun_.SetParticlePosition(G4ThreeVector{0, 0, 0});          // origin
    gun_.SetParticleMomentumDirection(G4ThreeVector{1, 0, 0}); // +x
}

//---------------------------------------------------------------------------//
/*!
 * Generate primaries based on the particle gun data.
 */
void PrimaryGeneratorAction::GeneratePrimaries(G4Event* event)
{
    gun_.GeneratePrimaryVertex(event);
}

//---------------------------------------------------------------------------//
} // namespace demo_geant
