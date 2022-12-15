//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-geant-integration/PrimaryGeneratorAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <G4ParticleGun.hh>
#include <G4VUserPrimaryGeneratorAction.hh>

#include "HepMC3Reader.hh"

namespace demo_geant
{
//---------------------------------------------------------------------------//
/*!
 * Minimal implementation of a primary generator action class.
 *
 * TODO: replace with HepMC3 input.
 */
class PrimaryGeneratorAction final : public G4VUserPrimaryGeneratorAction
{
  public:
    // Construct with default particle gun
    PrimaryGeneratorAction();

    // Construct with HepMC3 input file
    PrimaryGeneratorAction(std::shared_ptr<G4VPrimaryGenerator> hepmc3_reader);

    // Generate events
    void GeneratePrimaries(G4Event* event) final;

  private:
    std::unique_ptr<G4ParticleGun>       gun_{nullptr};
    std::shared_ptr<G4VPrimaryGenerator> hepmc3_reader_;
};

//---------------------------------------------------------------------------//
} // namespace demo_geant
