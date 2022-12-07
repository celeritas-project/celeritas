//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/PrimaryGeneratorAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <G4SystemOfUnits.hh>
#include <G4ThreeVector.hh>
#include <G4VUserPrimaryGeneratorAction.hh>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Minimal implementation of a primary generator action class.
 * Construct by providing basic particle gun information.
 */
class PrimaryGeneratorAction final : public G4VUserPrimaryGeneratorAction
{
  public:
    struct ParticleGun
    {
        int           pdg_id;
        double        energy;
        G4ThreeVector dir;
        G4ThreeVector pos;
    };

    // Construct by providing a particle gun setup
    PrimaryGeneratorAction(ParticleGun particle_gun);

    // Generate events
    void GeneratePrimaries(G4Event* event) final;

  private:
    ParticleGun particle_gun_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
