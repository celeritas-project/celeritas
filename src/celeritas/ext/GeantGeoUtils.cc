//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantGeoUtils.cc
//---------------------------------------------------------------------------//
#include "GeantGeoUtils.hh"

#include <iostream>
#include <string>
#include <G4GDMLParser.hh>
#include <G4LogicalVolume.hh>
#include <G4LogicalVolumeStore.hh>
#include <G4PhysicalVolumeStore.hh>
#include <G4SolidStore.hh>
#include <G4TouchableHistory.hh>
#include <G4VPhysicalVolume.hh>
#include <G4ios.hh>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/ScopedStreamRedirect.hh"
#include "corecel/sys/ScopedMem.hh"

#include "ScopedGeantExceptionHandler.hh"
#include "ScopedGeantLogger.hh"

namespace celeritas
{
namespace
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
G4VPhysicalVolume*
load_geant_geometry_impl(std::string const& filename, bool strip_pointer_ext)
{
    CELER_LOG(info) << "Loading Geant4 geometry from GDML at " << filename;
    ScopedMem record_mem("load_geant_geometry");

    {
        // I have no idea why, but creating the GDML parser resets the
        // ScopedGeantLogger on its first instantiation (geant4@11.0)
        G4GDMLParser temp_parser_init;
    }

    ScopedGeantLogger scoped_logger;
    ScopedGeantExceptionHandler scoped_exceptions;

    G4GDMLParser gdml_parser;
    // Note that material and element names (at least as
    // of Geant4@11.0) are *always* stripped: only volumes and solids keep
    // their extension.
    gdml_parser.SetStripFlag(strip_pointer_ext);

    gdml_parser.Read(filename, /* validate_gdml_schema = */ false);

    G4VPhysicalVolume* result(gdml_parser.GetWorldVolume());
    CELER_ENSURE(result);
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Print detailed information about the touchable history.
 */
std::ostream& operator<<(std::ostream& os, PrintableNavHistory const& pnh)
{
    CELER_EXPECT(pnh.touch);
    os << '{';

    G4VTouchable& touch = const_cast<G4VTouchable&>(*pnh.touch);
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
 * Print the logical volume name, ID, and address.
 */
std::ostream& operator<<(std::ostream& os, PrintableLV const& plv)
{
    if (plv.lv)
    {
        os << '"' << plv.lv->GetName() << "\"@"
           << static_cast<void const*>(plv.lv)
           << " (ID=" << plv.lv->GetInstanceID() << ')';
    }
    else
    {
        os << "{null G4LogicalVolume}";
    }
    return os;
}

//---------------------------------------------------------------------------//
/*!
 * Load a Geant4 geometry, leaving the pointer suffixes intact for VecGeom.
 *
 * Do *not* strip `0x` extensions since those are needed to deduplicate complex
 * geometries (e.g. CMS) when loaded separately by VGDML and Geant4. The
 * pointer-based deduplication is handled by the Label and LabelIdMultiMap.
 *
 * \return Geant4-owned world volume
 */
G4VPhysicalVolume* load_geant_geometry(std::string const& filename)
{
    return load_geant_geometry_impl(filename, false);
}

//---------------------------------------------------------------------------//
/*!
 * Load a Geant4 geometry, stripping suffixes like a typical Geant4 app.
 *
 * With this implementation, we let Geant4 strip the uniquifying pointers,
 * which allows our application to construct its own based on the actual
 * in-memory addresses.
 *
 * \return Geant4-owned world volume
 */
G4VPhysicalVolume* load_geant_geometry_native(std::string const& filename)
{
    return load_geant_geometry_impl(filename, true);
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
/*!
 * Get a view to the Geant4 LV store.
 *
 * This includes all volumes, potentially null ones as well.
 */
Span<G4LogicalVolume*> geant_logical_volumes()
{
    G4LogicalVolumeStore* lv_store = G4LogicalVolumeStore::GetInstance();
    CELER_ASSERT(lv_store);
    auto start = lv_store->data();
    auto stop = start + lv_store->size();
    while (start != stop && *start == nullptr)
    {
        ++start;
    }
    return {start, stop};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
