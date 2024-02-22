//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/GeantGeoUtils.cc
//---------------------------------------------------------------------------//
#include "GeantGeoUtils.hh"

#include <iostream>
#include <string>
#include <string_view>
#include <unordered_set>
#include <G4GDMLParser.hh>
#include <G4GDMLWriteStructure.hh>
#include <G4LogicalVolume.hh>
#include <G4LogicalVolumeStore.hh>
#include <G4PhysicalVolumeStore.hh>
#include <G4ReflectionFactory.hh>
#include <G4SolidStore.hh>
#include <G4Threading.hh>
#include <G4TouchableHistory.hh>
#include <G4VPhysicalVolume.hh>
#include <G4Version.hh>
#include <G4ios.hh>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/io/Join.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/ScopedStreamRedirect.hh"
#include "corecel/io/ScopedTimeLog.hh"
#include "corecel/sys/ScopedMem.hh"

#include "ScopedGeantExceptionHandler.hh"
#include "ScopedGeantLogger.hh"

// Check Geant4-reported and CMake-configured versions, mapping from
// Geant4's base-10 XXYZ -> to Celeritas base-16 0xXXYYZZ
static_assert(G4VERSION_NUMBER
                  == 100 * (CELERITAS_GEANT4_VERSION / 0x10000)
                         + 10 * ((CELERITAS_GEANT4_VERSION / 0x100) % 0x100)
                         + (CELERITAS_GEANT4_VERSION % 0x100),
              "CMake-reported Geant4 version does not match installed "
              "<G4Version.hh>: compare to 'celeritas_sys_config.h'");

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

    if (!G4Threading::IsMasterThread())
    {
        // Always-on debug assertion (not a "runtime" error but a
        // subtle programming logic error that always causes a crash)
        CELER_DEBUG_FAIL(
            "Geant4 geometry cannot be loaded from a worker thread", internal);
    }

    ScopedMem record_mem("load_geant_geometry");
    ScopedTimeLog scoped_time;

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

    auto& touch = const_cast<GeantTouchableBase&>(*pnh.touch);
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
        G4ReflectionFactory::Instance()->Clean();
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
/*!
 * Find Geant4 logical volumes corresponding to a list of names.
 *
 * If logical volumes with duplicate names are present, they will all show up
 * in the output and a warning will be emitted. If one is missing, a
 * \c RuntimeError will be raised.
 *
 * \code
   static std::string_view const labels[] = {"Vol1", "Vol2"};
   auto vols = find_geant_volumes(make_span(labels));
 * \endcode
 */
std::unordered_set<G4LogicalVolume const*>
find_geant_volumes(std::unordered_set<std::string> names)
{
    // Find all names that match the set
    std::unordered_set<G4LogicalVolume const*> result;
    result.reserve(names.size());
    for (auto* lv : geant_logical_volumes())
    {
        if (lv && names.count(lv->GetName()))
        {
            result.insert(lv);
        }
    }

    // Remove found names and warn about duplicates
    for (auto* lv : result)
    {
        auto iter = names.find(lv->GetName());
        if (iter != names.end())
        {
            names.erase(iter);
        }
        else
        {
            CELER_LOG(warning)
                << "Multiple Geant4 volumes are mapped to name '"
                << lv->GetName();
        }
    }

    // Make sure all requested names are found
    CELER_VALIDATE(names.empty(),
                   << "failed to find Geant4 volumes corresponding to the "
                      "following names: "
                   << join(names.begin(), names.end(), ", "));

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Generate the GDML name for a Geant4 logical volume.
 */
std::string make_gdml_name(G4LogicalVolume const& lv)
{
    // Run the LV through the GDML export name generator so that the volume is
    // uniquely identifiable in VecGeom. Reuse the same instance to reduce
    // overhead: note that the method isn't const correct.
    static G4GDMLWriteStructure temp_writer;

    auto const* refl_factory = G4ReflectionFactory::Instance();
    if (auto const* unrefl_lv
        = refl_factory->GetConstituentLV(const_cast<G4LogicalVolume*>(&lv)))
    {
        // If this is a reflected volume, add the reflection extension after
        // the final pointer to match the converted VecGeom name
        std::string name
            = temp_writer.GenerateName(unrefl_lv->GetName(), unrefl_lv);
        name += refl_factory->GetVolumesNameExtension();
        return name;
    }

    return temp_writer.GenerateName(lv.GetName(), &lv);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
