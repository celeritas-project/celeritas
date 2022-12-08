//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-geant-integration/ActionInitialization.hh
//---------------------------------------------------------------------------//
#pragma once

#include <G4VUserActionInitialization.hh>

#include "PrimaryGeneratorAction.hh"

namespace demo_geant
{
//---------------------------------------------------------------------------//
/*!
 * Set up demo-geant-integration specific action initializations.
 */
class ActionInitialization final : public G4VUserActionInitialization
{
  public:
    //!@{
    //! \name Type aliases
    using PGAParticleGun = PrimaryGeneratorAction::ParticleGun;
    //!@}

  public:
    // Construct with default particle gun
    ActionInitialization();

    // Construct with user-defined particle gun
    ActionInitialization(PGAParticleGun particle_gun);

    void BuildForMaster() const final;
    void Build() const final;

  private:
    PGAParticleGun particle_gun_;
};

//---------------------------------------------------------------------------//
} // namespace demo_geant
