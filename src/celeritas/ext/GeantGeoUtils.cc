//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantGeoUtils.cc
//---------------------------------------------------------------------------//
#include "GeantGeoUtils.hh"

#include <G4GDMLParser.hh>
#include <G4LogicalVolumeStore.hh>
#include <G4PhysicalVolumeStore.hh>
#include <G4SolidStore.hh>
#include <G4VPhysicalVolume.hh>

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/ScopedTimeAndRedirect.hh"
#include "corecel/sys/ScopedMem.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Load a gdml input file, creating a pointer owned by Geant4.
 *
 * Geant4's constructors for physical/logical volumes register \c this pointers
 * in a vector which is cleaned up manually. Yuck.
 *
 * \return the world volume
 */
G4VPhysicalVolume* load_geant_geometry(std::string const& filename)
{
    CELER_LOG(info) << "Loading Geant4 geometry from GDML at " << filename;
    ScopedMem record_mem("load_geant_geometry");

    // Create parser; do *not* strip `0x` extensions since those are needed to
    // deduplicate complex geometries (e.g. CMS) and are handled by the Label
    // and LabelIdMultiMap. Note that material and element names (at least as
    // of Geant4@11.0) are *always* stripped: only volumes and solids keep
    // their extension.
    G4GDMLParser gdml_parser;
    gdml_parser.SetStripFlag(false);

    gdml_parser.Read(filename, /* validate_gdml_schema = */ false);

    G4VPhysicalVolume* result(gdml_parser.GetWorldVolume());
    CELER_ENSURE(result);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Reset all Geant4 geometry stores if *not* using RunManager.
 *
 * Use this function if reading geometry and cleaning up *without* doing any
 * transport in between (useful for geometry conversion testing).
 */
void reset_geant_geometry()
{
    CELER_LOG(debug) << "Resetting Geant4 geometry stores";

    G4PhysicalVolumeStore::Clean();
    G4LogicalVolumeStore::Clean();
    G4SolidStore::Clean();
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
