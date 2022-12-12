//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-geant-integration/PrimaryGeneratorAction.cc
//---------------------------------------------------------------------------//
#include "PrimaryGeneratorAction.hh"

#include <G4ParticleGun.hh>
#include <G4ParticleTable.hh>
#include <G4SystemOfUnits.hh>
#include <G4ThreeVector.hh>

#include "corecel/io/Logger.hh"

namespace demo_geant
{
//---------------------------------------------------------------------------//
/*!
 * Generate primaries based on the particle gun data.
 */
void PrimaryGeneratorAction::GeneratePrimaries(G4Event* event)
{
    CELER_LOG_LOCAL(debug) << "PrimaryGeneratorAction::GeneratePrimaries";

    auto g4particle_def = G4ParticleTable::GetParticleTable()->FindParticle(11);

    G4ParticleGun g4particle_gun;
    g4particle_gun.SetParticleDefinition(g4particle_def);
    g4particle_gun.SetParticleEnergy(500 * MeV);
    g4particle_gun.SetParticlePosition(G4ThreeVector{0, 0, 0}); // origin
    g4particle_gun.SetParticleMomentumDirection(G4ThreeVector{1, 0, 0}); // +x
    g4particle_gun.GeneratePrimaryVertex(event);
}

//---------------------------------------------------------------------------//
} // namespace demo_geant
