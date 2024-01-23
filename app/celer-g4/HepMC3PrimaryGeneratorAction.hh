//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/HepMC3PrimaryGeneratorAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <G4VUserPrimaryGeneratorAction.hh>

namespace celeritas
{
class HepMC3PrimaryGenerator;
namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Generate events by reading from a HepMC3 file.
 */
class HepMC3PrimaryGeneratorAction final : public G4VUserPrimaryGeneratorAction
{
  public:
    //!@{
    //! \name Type aliases
    using SPGenerator = std::shared_ptr<HepMC3PrimaryGenerator>;
    //!@}

  public:
    // Construct from a shared generator
    explicit HepMC3PrimaryGeneratorAction(SPGenerator);

    // Generate events
    void GeneratePrimaries(G4Event* event) final;

  private:
    SPGenerator generator_;
};

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
