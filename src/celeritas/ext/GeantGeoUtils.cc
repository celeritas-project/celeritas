//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantGeoUtils.cc
//---------------------------------------------------------------------------//
#include "GeantGeoUtils.hh"

#include <iostream>
#include <G4GDMLParser.hh>
#include <G4LogicalVolume.hh>
#include <G4LogicalVolumeStore.hh>
#include <G4PhysicalVolumeStore.hh>
#include <G4SolidStore.hh>
#include <G4TouchableHistory.hh>
#include <G4VPhysicalVolume.hh>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/ScopedStreamRedirect.hh"
#include "corecel/io/ScopedTimeAndRedirect.hh"
#include "corecel/sys/ScopedMem.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Print detailed information about the touchable history.
 */
std::ostream& operator<<(std::ostream& os, PrintableNavHistory const& pnh)
{
    CELER_EXPECT(pnh.touch);
    os << '{';

    G4VTouchable& touch = *pnh.touch;
    for (int depth : range(touch.GetHistoryDepth()))
    {
        G4VPhysicalVolume* vol = touch.GetVolume(depth);
        CELER_ASSERT(vol);
        G4LogicalVolume* lv = vol->GetLogicalVolume();
        CELER_ASSERT(lv);
        if (depth != 0)
        {
            os << " -> ";
        }
        os << "{pv='" << vol->GetName() << "', lv=" << lv->GetInstanceID()
           << "='" << lv->GetName() << "'}";
    }
    os << '}';
    return os;
}

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
    ScopedTimeAndRedirect scoped_time{"G4GDMLParser"};
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

    std::string msg;
    {
        ScopedStreamRedirect scoped_log(&std::cout);

        G4PhysicalVolumeStore::Clean();
        G4LogicalVolumeStore::Clean();
        G4SolidStore::Clean();
        msg = scoped_log.str();
    }
    if (!msg.empty())
    {
        CELER_LOG(diagnostic) << "While closing Geant4 geometry: " << msg;
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
