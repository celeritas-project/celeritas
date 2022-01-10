//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file DetectorConstruction.cc
//---------------------------------------------------------------------------//
#include "DetectorConstruction.hh"

#include <G4GDMLParser.hh>

#include "base/Assert.hh"
#include "comm/Logger.hh"

namespace geant_exporter
{
//---------------------------------------------------------------------------//
/*!
 * Construct with a given gdml input file.
 */
DetectorConstruction::DetectorConstruction(G4String gdmlInput)
{
    CELER_LOG(info) << "Loading geometry from " << gdmlInput;
    G4GDMLParser   gdml_parser;
    constexpr bool validate_gdml_schema = false;
    gdml_parser.Read(gdmlInput, validate_gdml_schema);
    phys_vol_world_.reset(gdml_parser.GetWorldVolume());
    CELER_ENSURE(phys_vol_world_);
}

//---------------------------------------------------------------------------//
/*!
 * Default destructor.
 */
DetectorConstruction::~DetectorConstruction() = default;

//---------------------------------------------------------------------------//
/*!
 * Return the loaded world volume.
 */
G4VPhysicalVolume* DetectorConstruction::Construct()
{
    CELER_EXPECT(phys_vol_world_);
    return phys_vol_world_.release();
}

//---------------------------------------------------------------------------//
/*!
 * Return the world physical volume pointer.
 *
 * This must be called before Construct() is invoked, i.e. before releasing
 * unique_ptr<DetectorConstruction> to the run manager in main().
 */
const G4VPhysicalVolume* DetectorConstruction::get_world_volume() const
{
    CELER_EXPECT(phys_vol_world_);
    return phys_vol_world_.get();
}

//---------------------------------------------------------------------------//
} // namespace geant_exporter
