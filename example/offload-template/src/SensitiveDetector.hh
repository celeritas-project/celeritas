//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file offload-template/src/SensitiveDetector.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include <G4VSensitiveDetector.hh>

//---------------------------------------------------------------------------//
/*!
 * Sensitive detector class.
 *
 * This is currently the *only* interface between Geant4 and Celeritas.
 */
class SensitiveDetector : public G4VSensitiveDetector
{
  public:
    // Construct with SD name
    SensitiveDetector(std::string sd_name);

    // Celeritas callback interface
    G4bool ProcessHits(G4Step*, G4TouchableHistory*) final;
};
