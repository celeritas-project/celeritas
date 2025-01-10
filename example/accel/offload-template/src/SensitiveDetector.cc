//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file example/accel/offload-template/src/SensitiveDetector.cc
//---------------------------------------------------------------------------//
#include "SensitiveDetector.hh"

//---------------------------------------------------------------------------//
/*!
 * Construct with sensitive detector name.
 */
SensitiveDetector::SensitiveDetector(std::string sd_name)
    : G4VSensitiveDetector(sd_name)
{
}

//---------------------------------------------------------------------------//
/*!
 * Callback interface with Celeritas.
 */
G4bool SensitiveDetector::ProcessHits(G4Step*, G4TouchableHistory*)
{
    // -----------------------------------------------------
    // Only existing interface between Celeritas and Geant4
    // -----------------------------------------------------
    // Data processed through other methods (e.g. SteppingAction) will not be
    // correctly passed to the I/O during an offloaded run.
    return true;
}
