//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/PrimaryGeneratorAction.cc
//---------------------------------------------------------------------------//
#include "PrimaryGeneratorAction.hh"

#include <G4ParticleTable.hh>
#include <G4SystemOfUnits.hh>

#include "corecel/Assert.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct a particle gun for the minimal simulation run.
 */
PrimaryGeneratorAction::PrimaryGeneratorAction()
    : G4VUserPrimaryGeneratorAction(), particle_gun_(nullptr)
{
    // Select particle type
    G4ParticleDefinition* particle;
    particle = G4ParticleTable::GetParticleTable()->FindParticle("e-");
    CELER_ASSERT(particle);

    // Create and set up particle gun
    const int     number_of_particles = 1;
    G4ThreeVector pos(0, 0, 0);
    particle_gun_ = std::make_unique<G4ParticleGun>(number_of_particles);
    particle_gun_->SetParticleDefinition(particle);
    particle_gun_->SetParticleMomentumDirection(G4ThreeVector(0., 0., 1.));
    particle_gun_->SetParticleEnergy(10 * GeV);
    particle_gun_->SetParticlePosition(pos);
}

//---------------------------------------------------------------------------//
/*!
 * Default destructor.
 */
PrimaryGeneratorAction::~PrimaryGeneratorAction() = default;

//---------------------------------------------------------------------------//
/*!
 * Generate primary particle at the beginning of each event.
 */
void PrimaryGeneratorAction::GeneratePrimaries(G4Event* event)
{
    particle_gun_->GeneratePrimaryVertex(event);
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
