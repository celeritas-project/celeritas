//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file offload-template/src/PrimaryGeneratorAction.cc
//---------------------------------------------------------------------------//
#include "PrimaryGeneratorAction.hh"

#include <G4ParticleGun.hh>
#include <G4ParticleTable.hh>
#include <G4SystemOfUnits.hh>

#include "DetectorConstruction.hh"

namespace celeritas
{
namespace example
{
//---------------------------------------------------------------------------//
/*!
 * Construct empty.
 */
PrimaryGeneratorAction::PrimaryGeneratorAction()
    : G4VUserPrimaryGeneratorAction()
{
}

//---------------------------------------------------------------------------//
/*!
 * Generate a simple primary.
 */
void PrimaryGeneratorAction::GeneratePrimaries(G4Event* event)
{
    G4ParticleGun particle_gun;
    particle_gun.SetParticleDefinition(
        G4ParticleTable::GetParticleTable()->FindParticle(11));  // e-
    particle_gun.SetParticleEnergy(1 * MeV);
    particle_gun.SetParticlePosition(G4ThreeVector());  // Origin
    particle_gun.SetParticleMomentumDirection(G4ThreeVector(1, 0, 0));  // +x
    particle_gun.GeneratePrimaryVertex(event);
}

}  // namespace example
}  // namespace celeritas
