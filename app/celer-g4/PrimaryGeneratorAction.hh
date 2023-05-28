//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-geant-integration/PrimaryGeneratorAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <G4VUserPrimaryGeneratorAction.hh>

namespace demo_geant
{
//---------------------------------------------------------------------------//
/*!
 * Generate events by reading from a HepMC3 file.
 */
class PrimaryGeneratorAction final : public G4VUserPrimaryGeneratorAction
{
  public:
    // Get the total number of events available in the HepMC3 file
    static int NumEvents();

    // Construct primary action
    PrimaryGeneratorAction() = default;

    // Generate events
    void GeneratePrimaries(G4Event* event) final;
};

//---------------------------------------------------------------------------//
}  // namespace demo_geant
