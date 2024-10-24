//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/g4/GeantGeoParams.cc
//---------------------------------------------------------------------------//
#include "GeantGeoParams.hh"

#include <vector>
#include <G4GeometryManager.hh>
#include <G4LogicalVolume.hh>
#include <G4LogicalVolumeStore.hh>
#include <G4PhysicalVolumeStore.hh>
#include <G4Transportation.hh>
#include <G4TransportationManager.hh>
#include <G4VPhysicalVolume.hh>
#include <G4VSolid.hh>
#include <G4Version.hh>
#include <G4VisExtent.hh>
#if G4VERSION_NUMBER >= 1070
#    include <G4Backtrace.hh>
#endif

#include "corecel/Config.hh"

#include "corecel/cont/Range.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/StringUtils.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/ScopedMem.hh"
#include "geocel/GeantGeoUtils.hh"
#include "geocel/GeantUtils.hh"
#include "geocel/ScopedGeantExceptionHandler.hh"
#include "geocel/ScopedGeantLogger.hh"

#include "Convert.hh"  // IWYU pragma: associated
#include "GeantGeoData.hh"  // IWYU pragma: associated
#include "VisitGeantVolumes.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
std::vector<Label>
get_volume_labels(G4VPhysicalVolume const& world, bool unique_volumes)
{
    std::vector<Label> labels;
    visit_geant_volumes(
        [&](G4LogicalVolume const& lv) {
            auto i = static_cast<std::size_t>(lv.GetInstanceID());
            if (i >= labels.size())
            {
                labels.resize(i + 1);
            }
            if (unique_volumes)
            {
                labels[i] = Label::from_geant(make_gdml_name(lv));
            }
            else
            {
                labels[i] = Label::from_geant(lv.GetName());
            }
        },
        world);
    return labels;
}

//---------------------------------------------------------------------------//
std::vector<Label> get_pv_labels(G4VPhysicalVolume const& world)
{
    std::vector<Label> labels;
    std::unordered_map<G4VPhysicalVolume const*, int> max_depth;

    visit_geant_volume_instances(
        [&labels, &max_depth](G4VPhysicalVolume const& pv, int depth) {
            auto&& [iter, inserted] = max_depth.insert({&pv, depth});
            if (!inserted)
            {
                if (iter->second >= depth)
                {
                    // Already visited PV at this depth or more
                    return false;
                }
                // Update the max depth
                iter->second = depth;
            }

            auto i = static_cast<std::size_t>(pv.GetInstanceID());
            if (i >= labels.size())
            {
                labels.resize(i + 1);
            }
            if (labels[i].empty())
            {
                labels[i] = Label::from_geant(pv.GetName());
                CELER_ASSERT(!labels[i].empty());
            }
            return true;
        },
        world);
    return labels;
}

//---------------------------------------------------------------------------//
LevelId::size_type get_max_depth(G4VPhysicalVolume const& world)
{
    int result{0};
    visit_geant_volume_instances(
        [&result](G4VPhysicalVolume const&, int level) {
            result = max(level, result);
            return true;
        },
        world);
    CELER_ENSURE(result >= 0);
    // Maximum "depth" is one greater than "highest level"
    return static_cast<LevelId::size_type>(result + 1);
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct from a GDML input.
 *
 * This assumes that Celeritas is driving and will manage Geant4 exceptions
 * etc.
 */
GeantGeoParams::GeantGeoParams(std::string const& filename)
{
    ScopedMem record_mem("GeantGeoParams.construct");

    disable_geant_signal_handler();

    if (!ends_with(filename, ".gdml"))
    {
        CELER_LOG(warning) << "Expected '.gdml' extension for GDML input";
    }

    host_ref_.world = load_geant_geometry(filename);
    loaded_gdml_ = true;

    // NOTE: only instantiate the logger/exception handler *after* loading
    // Geant4 geometry, since something in the GDML parser's call chain resets
    // the Geant logger.
    scoped_logger_ = std::make_unique<ScopedGeantLogger>();
    scoped_exceptions_ = std::make_unique<ScopedGeantExceptionHandler>();

    this->build_tracking();
    this->build_metadata();

    CELER_ENSURE(volumes_);
    CELER_ENSURE(host_ref_);
}

//---------------------------------------------------------------------------//
/*!
 * Use an existing loaded Geant4 geometry.
 */
GeantGeoParams::GeantGeoParams(G4VPhysicalVolume const* world)
{
    CELER_EXPECT(world);
    host_ref_.world = const_cast<G4VPhysicalVolume*>(world);

    ScopedMem record_mem("GeantGeoParams.construct");

    // Verify consistency of the world volume
    G4VPhysicalVolume const* nav_world = [] {
        auto* man = G4TransportationManager::GetTransportationManager();
        CELER_ASSERT(man);
        auto* nav = man->GetNavigatorForTracking();
        CELER_ENSURE(nav);
        return nav->GetWorldVolume();
    }();
    if (world != nav_world)
    {
        auto msg = CELER_LOG(warning);
        msg << "Geant4 geometry was initialized with inconsistent "
               "world volume: given '"
            << world->GetName() << "'@' " << static_cast<void const*>(world)
            << "; navigation world is ";
        if (nav_world)
        {
            msg << nav_world->GetName() << "'@' "
                << static_cast<void const*>(nav_world);
        }
        else
        {
            msg << "unset";
        }
    }

    this->build_tracking();
    this->build_metadata();

    CELER_ENSURE(volumes_);
    CELER_ENSURE(host_ref_);
}

//---------------------------------------------------------------------------//
/*!
 * Clean up on destruction.
 */
GeantGeoParams::~GeantGeoParams()
{
    if (closed_geometry_)
    {
        G4GeometryManager::GetInstance()->OpenGeometry(host_ref_.world);
    }
    if (loaded_gdml_)
    {
        reset_geant_geometry();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Locate the volume ID corresponding to a Geant4 logical volume.
 */
VolumeId GeantGeoParams::find_volume(G4LogicalVolume const* volume) const
{
    CELER_EXPECT(volume);
    auto result = id_cast<VolumeId>(volume->GetInstanceID());
    if (!(result < volumes_.size()))
    {
        // Volume is out of range: possibly an LV defined after this geometry
        // class was created
        result = {};
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Get the Geant4 physical volume corresponding to a volume instance ID.
 *
 * If the input ID is false, a null pointer will be returned.
 */
G4VPhysicalVolume const* GeantGeoParams::id_to_pv(VolumeInstanceId id) const
{
    CELER_EXPECT(!id || id < vol_instances_.size());
    if (!id)
    {
        return nullptr;
    }

    G4PhysicalVolumeStore* pv_store = G4PhysicalVolumeStore::GetInstance();
    auto index = id.unchecked_get() - pv_offset_;
    CELER_ASSERT(index < pv_store->size());
    return (*pv_store)[index];
}

//---------------------------------------------------------------------------//
/*!
 * Get the Geant4 logical volume corresponding to a volume ID.
 *
 * If the input volume ID is unassigned, a null pointer will be returned.
 */
G4LogicalVolume const* GeantGeoParams::id_to_lv(VolumeId id) const
{
    CELER_EXPECT(!id || id < volumes_.size());
    if (!id)
    {
        return nullptr;
    }

    G4LogicalVolumeStore* lv_store = G4LogicalVolumeStore::GetInstance();
    auto index = id.unchecked_get() - lv_offset_;
    CELER_ASSERT(index < lv_store->size());
    return (*lv_store)[index];
}

//---------------------------------------------------------------------------//
/*!
 * Complete geometry construction
 */
void GeantGeoParams::build_tracking()
{
    // Close the geometry if needed
    auto* geo_man = G4GeometryManager::GetInstance();
    CELER_ASSERT(geo_man);
    if (!geo_man->IsGeometryClosed())
    {
        geo_man->CloseGeometry(
            /* optimize = */ true, /* verbose = */ false, host_ref_.world);
        closed_geometry_ = true;
    }
}

//---------------------------------------------------------------------------//
/*!
 * Construct Celeritas host-only metadata.
 */
void GeantGeoParams::build_metadata()
{
    CELER_EXPECT(host_ref_);

    ScopedMem record_mem("GeantGeoParams.build_metadata");

    // Get offset of logical/physical volumes present in unit tests
    lv_offset_ = [] {
        G4LogicalVolumeStore* lv_store = G4LogicalVolumeStore::GetInstance();
        CELER_ASSERT(lv_store && !lv_store->empty());
        return lv_store->front()->GetInstanceID();
    }();
    pv_offset_ = [] {
        G4PhysicalVolumeStore* pv_store = G4PhysicalVolumeStore::GetInstance();
        CELER_ASSERT(pv_store && !pv_store->empty());
        return pv_store->front()->GetInstanceID();
    }();

    // Construct volume labels
    volumes_ = VolumeMap{"volume",
                         get_volume_labels(*host_ref_.world, !loaded_gdml_)};
    vol_instances_
        = VolInstanceMap{"volume instance", get_pv_labels(*host_ref_.world)};
    max_depth_ = get_max_depth(*host_ref_.world);

    // Save world bbox (NOTE: assumes no transformation on PV?)
    bbox_ = [world_lv = host_ref_.world->GetLogicalVolume()] {
        CELER_EXPECT(world_lv);
        G4VSolid const* solid = world_lv->GetSolid();
        CELER_ASSERT(solid);
        G4VisExtent const& extent = solid->GetExtent();

        return BBox({convert_from_geant(G4ThreeVector(extent.GetXmin(),
                                                      extent.GetYmin(),
                                                      extent.GetZmin()),
                                        clhep_length),
                     convert_from_geant(G4ThreeVector(extent.GetXmax(),
                                                      extent.GetYmax(),
                                                      extent.GetZmax()),
                                        clhep_length)});
    }();
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
