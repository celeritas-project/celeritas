//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
#include "geocel/GeantGdmlLoader.hh"
#include "geocel/GeantGeoUtils.hh"
#include "geocel/GeantUtils.hh"
#include "geocel/ScopedGeantExceptionHandler.hh"
#include "geocel/ScopedGeantLogger.hh"

#include "Convert.hh"  // IWYU pragma: associated
#include "GeantGeoData.hh"  // IWYU pragma: associated
#include "VisitVolumes.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
LevelId::size_type get_max_depth(G4VPhysicalVolume const& world)
{
    LevelId::size_type result{0};
    visit_volume_instances(
        [&result](G4VPhysicalVolume const&, int level) {
            result = max(level, static_cast<int>(result));
            return true;
        },
        world);
    // Maximum "depth" is one greater than "highest level"
    return result + 1;
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct from a GDML input.
 *
 * This assumes that Celeritas is driving and will manage Geant4 logging
 * and exceptions.
 */
GeantGeoParams::GeantGeoParams(std::string const& filename)
{
    ScopedMem record_mem("GeantGeoParams.construct");

    scoped_logger_
        = std::make_unique<ScopedGeantLogger>(celeritas::world_logger());
    scoped_exceptions_ = std::make_unique<ScopedGeantExceptionHandler>();

    disable_geant_signal_handler();

    if (!ends_with(filename, ".gdml"))
    {
        CELER_LOG(warning) << "Expected '.gdml' extension for GDML input";
    }

    host_ref_.world = load_gdml(filename);
    loaded_gdml_ = true;

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
    G4VPhysicalVolume const* nav_world = geant_world_volume();
    if (world != nav_world)
    {
        auto msg = CELER_LOG(warning);
        msg << "Geant4 geometry was initialized with inconsistent "
               "world volume: given '"
            << world->GetName() << "'@" << static_cast<void const*>(world)
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
 * \warning For Geant4 parameterised/replicated volumes, external state (e.g.
 * the local navigation) \em must be used in concert with this method: i.e.,
 * navigation on the current thread needs to have just "visited" the instance.
 *
 * \todo Create our own volume instance mapping for Geant4, where
 * VolumeInstanceId corresponds to a (G4PV*, int) pair rather than just G4PV*
 * (which in Geant4 is not sufficiently unique).
 */
GeantPhysicalInstance GeantGeoParams::id_to_geant(VolumeInstanceId id) const
{
    CELER_EXPECT(!id || id < vol_instances_.size());
    if (!id)
    {
        return {};
    }

    G4PhysicalVolumeStore* pv_store = G4PhysicalVolumeStore::GetInstance();
    auto index = id.unchecked_get() - pv_offset_;
    CELER_ASSERT(index < pv_store->size());

    GeantPhysicalInstance result;
    result.pv = (*pv_store)[index];
    CELER_ASSERT(result.pv);
    if (result.pv->VolumeType() != EVolume::kNormal)
    {
        auto copy_no = result.pv->GetCopyNo();
        // NOTE: if this line fails, Geant4 may be returning uninitialized
        // memory on the local thread.
        CELER_ASSERT(copy_no >= 0 && copy_no < result.pv->GetMultiplicity());
        result.replica = id_cast<GeantPhysicalInstance::ReplicaId>(copy_no);
    }

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Get the Geant4 logical volume corresponding to a volume ID.
 *
 * If the input volume ID is unassigned, a null pointer will be returned.
 */
G4LogicalVolume const* GeantGeoParams::id_to_geant(VolumeId id) const
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
    if (lv_offset_ != 0 || pv_offset_ != 0)
    {
        CELER_LOG(debug) << "Building after volume stores were cleared: "
                         << "lv_offset=" << lv_offset_
                         << ", pv_offset=" << pv_offset_;
    }

    // Construct volume labels
    volumes_ = VolumeMap{"volume", make_logical_vol_labels(*host_ref_.world)};
    vol_instances_ = VolInstanceMap{
        "volume instance", make_physical_vol_labels(*host_ref_.world)};
    max_depth_ = get_max_depth(*host_ref_.world);

    // Save world bbox (NOTE: assumes no transformation on PV)
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
