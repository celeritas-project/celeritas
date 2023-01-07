//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
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
 * Minimal implementation of a primary generator action class.
 */
class PrimaryGeneratorAction final : public G4VUserPrimaryGeneratorAction
{
  public:
    // Construct primary action
    PrimaryGeneratorAction() = default;

    // Generate events
    void GeneratePrimaries(G4Event* event) final;
};

//---------------------------------------------------------------------------//
} // namespace demo_geant
