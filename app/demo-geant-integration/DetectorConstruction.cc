//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-geant-integration/DetectorConstruction.cc
//---------------------------------------------------------------------------//
#include "DetectorConstruction.hh"

#include "corecel/io/Logger.hh"

namespace demo_geant
{
//---------------------------------------------------------------------------//
G4VPhysicalVolume* DetectorConstruction::Construct()
{
    CELER_LOG_LOCAL(debug) << "DetectorConstruction::Construct";
    return world_.release();
}

//---------------------------------------------------------------------------//
void DetectorConstruction::ConstructSDandField()
{
    CELER_LOG_LOCAL(debug) << "DetectorConstruction::ConstructSDandField";
}

//---------------------------------------------------------------------------//
} // namespace demo_geant
