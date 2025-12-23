//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/GeantGeoUtils.cc
//---------------------------------------------------------------------------//
#include "GeantGeoUtils.hh"

#include <algorithm>
#include <iostream>
#include <string>
#include <unordered_set>
#include <G4Element.hh>
#include <G4FieldManager.hh>
#include <G4Isotope.hh>
#include <G4LogicalBorderSurface.hh>
#include <G4LogicalSkinSurface.hh>
#include <G4LogicalVolume.hh>
#include <G4LogicalVolumeStore.hh>
#include <G4Material.hh>
#include <G4NavigationHistory.hh>
#include <G4PhysicalVolumeStore.hh>
#include <G4ReflectionFactory.hh>
#include <G4RegionStore.hh>
#include <G4SolidStore.hh>
#include <G4Threading.hh>
#include <G4TouchableHistory.hh>
#include <G4TransportationManager.hh>
#include <G4VPhysicalVolume.hh>
#include <G4Version.hh>
#include <G4ios.hh>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/io/Join.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/ScopedStreamRedirect.hh"

// Check Geant4-reported and CMake-configured versions, mapping from
// Geant4's base-10 XXYZ -> to Celeritas base-16 0xXXYYZZ
static_assert(G4VERSION_NUMBER
                  == 100 * (CELERITAS_GEANT4_VERSION / 0x10000)
                         + 10 * ((CELERITAS_GEANT4_VERSION / 0x100) % 0x100)
                         + (CELERITAS_GEANT4_VERSION % 0x100),
              "CMake-reported Geant4 version does not match installed "
              "<G4Version.hh>: compare to 'corecel/Config.hh'");

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Free all pointers in a table.
 *
 * Geant4 requires "new"ing and *not* "delete"ing classes whose new/delete
 * operators modify an entry in a global table.
 */
template<class T>
void free_and_clear(std::vector<T*>* table)
{
    for (auto* ptr : *table)
    {
        delete ptr;
    }
    CELER_ASSERT(std::all_of(
        table->begin(), table->end(), [](T* ptr) { return ptr == nullptr; }));
    table->clear();
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Print detailed information about the touchable history.
 *
 * For brevity, this does not print the world volume.
 */
std::ostream& operator<<(std::ostream& os, StreamableNavHistory const& pnh)
{
    CELER_EXPECT(pnh.nav);
    os << '{';

    int depth = pnh.nav->GetDepth();
    for (int level : range(depth))
    {
        G4VPhysicalVolume* vol = pnh.nav->GetVolume(depth - level);
        CELER_ASSERT(vol);
        G4LogicalVolume* lv = vol->GetLogicalVolume();
        CELER_ASSERT(lv);
        if (level != 0)
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
std::ostream& operator<<(std::ostream& os, StreamableLV const& plv)
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
        G4RegionStore::Clean();
        G4SolidStore::Clean();
#if G4VERSION_NUMBER >= 1100
        G4ReflectionFactory::Instance()->Clean();
#endif
        G4LogicalSkinSurface::CleanSurfaceTable();
        G4LogicalBorderSurface::CleanSurfaceTable();
        free_and_clear(G4Material::GetMaterialTable());
        free_and_clear(const_cast<std::vector<G4Element*>*>(
            G4Element::GetElementTable()));
        free_and_clear(const_cast<std::vector<G4Isotope*>*>(
            G4Isotope::GetIsotopeTable()));
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
 * Get the world volume for the primary geometry.
 *
 * \return World volume if geometry has been initialized, nullptr otherwise.
 */
G4VPhysicalVolume const* geant_world_volume()
{
    auto* man = G4TransportationManager::GetTransportationManager();
    CELER_ASSERT(man);
    auto* nav = man->GetNavigatorForTracking();
    if (!nav)
        return nullptr;
    return nav->GetWorldVolume();
}

//---------------------------------------------------------------------------//
/*!
 * Get an optional global magnetic field for the tracking geometry.
 *
 * \return Field if geometry has been initialized and field exists, nullptr
 * otherwise.
 */
G4Field const* geant_field()
{
    auto* man = G4TransportationManager::GetTransportationManager();
    CELER_ASSERT(man);
    auto* field_mgr = man->GetFieldManager();
    if (!field_mgr)
        return nullptr;
    auto* detector_field = field_mgr->GetDetectorField();
    return detector_field;
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
                << lv->GetName() << "'";
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
}  // namespace celeritas
