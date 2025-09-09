//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file offload-template/src/SensitiveDetector.cc
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
 * Callback interface between Geant4 and Celeritas.
 */
G4bool SensitiveDetector::ProcessHits(G4Step*, G4TouchableHistory*)
{
    // -----------------------------------------------------
    // Only existing interface between Celeritas and Geant4
    // -----------------------------------------------------
    // Data processed through other classes (e.g. G4UserSteppingAction) will
    // not be correctly passed to the I/O during an offloaded run.
    //
    // See SetupOptions::SDSetupOptions (in Celeritas.cc) to enable the
    // necessary pre/post step attributes needed
    return true;
}
