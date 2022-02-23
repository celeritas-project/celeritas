//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file PrimaryGeneratorAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <G4Event.hh>
#include <G4ParticleGun.hh>
#include <G4VUserPrimaryGeneratorAction.hh>

namespace geant_exporter
{
//---------------------------------------------------------------------------//
/*!
 * Create a particle gun and generate one primary for a minimal simulation run.
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
