//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file PrimaryGeneratorAction.hh
//! \brief Generate primaries for a minimal simulation run
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include <G4VUserPrimaryGeneratorAction.hh>
#include <G4Event.hh>
#include <G4ParticleGun.hh>

namespace geant_exporter
{
//---------------------------------------------------------------------------//
/*!
 * Define the particle gun used in the Geant4 run.
 */
class PrimaryGeneratorAction : public G4VUserPrimaryGeneratorAction
{
  public:
    PrimaryGeneratorAction();
    ~PrimaryGeneratorAction();

    void GeneratePrimaries(G4Event* event) override;

  private:
    std::unique_ptr<G4ParticleGun> particle_gun_;
};

//---------------------------------------------------------------------------//
} // namespace geant_exporter
