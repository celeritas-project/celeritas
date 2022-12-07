//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/PrimaryGeneratorAction.cc
//---------------------------------------------------------------------------//
#include "PrimaryGeneratorAction.hh"

#include <G4ParticleGun.hh>
#include <G4ParticleTable.hh>

#include "corecel/io/Logger.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct a user-defined particle gun.
 */
PrimaryGeneratorAction::PrimaryGeneratorAction(ParticleGun particle_gun)
    : particle_gun_(particle_gun)
{
    CELER_LOG_LOCAL(debug) << "PrimaryGeneratorAction::PrimaryGeneratorAction "
                              "with user-defined particle gun";
}

//---------------------------------------------------------------------------//
/*!
 * Generate primaries based on the particle gun data.
 */
void PrimaryGeneratorAction::GeneratePrimaries(G4Event* event)
{
    CELER_LOG_LOCAL(debug) << "PrimaryGeneratorAction::GeneratePrimaries";

    auto g4particle_def = G4ParticleTable::GetParticleTable()->FindParticle(
        particle_gun_.pdg_id);

    G4ParticleGun g4particle_gun;
    g4particle_gun.SetParticleDefinition(g4particle_def);
    g4particle_gun.SetParticleEnergy(particle_gun_.energy);
    g4particle_gun.SetParticlePosition(particle_gun_.pos);
    g4particle_gun.SetParticleMomentumDirection(particle_gun_.dir.unit());
    g4particle_gun.GeneratePrimaryVertex(event);
}
//---------------------------------------------------------------------------//
} // namespace celeritas
