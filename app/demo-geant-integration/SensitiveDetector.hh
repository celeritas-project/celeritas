//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-geant-integration/SensitiveDetector.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include <G4VSensitiveDetector.hh>

namespace demo_geant
{
//---------------------------------------------------------------------------//
/*!
 * Example sensitive detector.
 */
class SensitiveDetector final : public G4VSensitiveDetector
{
  public:
    explicit SensitiveDetector(std::string name);

  protected:
    void   Initialize(G4HCofThisEvent*) final;
    G4bool ProcessHits(G4Step*, G4TouchableHistory*) final;
};

//---------------------------------------------------------------------------//
} // namespace demo_geant
