//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/PGPrimaryGeneratorAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <G4Event.hh>
#include <G4ParticleDefinition.hh>
#include <G4ParticleGun.hh>
#include <G4VUserPrimaryGeneratorAction.hh>

#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "celeritas/phys/PrimaryGenerator.hh"

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Generate events from a particle gun.
 *
 * This generates primary particles with energy, position, and direction
 * sampled from distributions specified by the user in \c
 * PrimaryGeneratorOptions.
 *
 * \sa PrimaryGenerator
 */
class PGPrimaryGeneratorAction final : public G4VUserPrimaryGeneratorAction
{
  public:
    //!@{
    //! \name Type aliases
    using Input = inp::CorePrimaryGenerator;
    //!@}

  public:
    // Construct primary action
    explicit PGPrimaryGeneratorAction(Input const& opts);

    // Generate events
    void GeneratePrimaries(G4Event* event) final;

  private:
    using GeneratorImpl = celeritas::PrimaryGenerator;

    std::vector<PDGNumber> pdg_;
    GeneratorImpl generate_primaries_;
    G4ParticleGun gun_;
};

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
