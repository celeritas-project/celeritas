//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/GeantGeoParams.cc
//---------------------------------------------------------------------------//
#include "GeantGeoParams.hh"

#include <unordered_map>
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

#include "GeantGdmlLoader.hh"
#include "GeantGeoUtils.hh"
#include "GeantUtils.hh"
#include "g4/Convert.hh"  // IWYU pragma: associated
#include "g4/GeantGeoData.hh"  // IWYU pragma: associated
#include "g4/VisitVolumes.hh"

#include "detail/MakeLabelVector.hh"

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
/*!
 * Get a reproducible vector of LV instance ID -> label from the given world.
 */
std::vector<Label> make_logical_vol_labels(G4VPhysicalVolume const& world)
{
    std::unordered_map<std::string, std::vector<G4LogicalVolume const*>> names;

    visit_volumes(
        [&](G4LogicalVolume const& lv) {
            std::string name = lv.GetName();
            if (name.empty())
            {
                CELER_LOG(debug)
                    << "Empty name for reachable LV id=" << lv.GetInstanceID();
                name = "[UNTITLED]";
            }
            // Add to name map
            names[std::move(name)].push_back(&lv);
        },
        world);

    return detail::make_label_vector<G4LogicalVolume const*>(
        std::move(names),
        [](G4LogicalVolume const* lv) { return lv->GetInstanceID(); });
}

//---------------------------------------------------------------------------//
/*!
 * Get a reproducible vector of PV instance ID -> label from the given world.
 */
std::vector<Label> make_physical_vol_labels(G4VPhysicalVolume const& world)
{
    std::unordered_map<G4VPhysicalVolume const*, int> max_depth;
    std::unordered_map<std::string, std::vector<G4VPhysicalVolume const*>> names;

    // Visit PVs, mapping names to instances, skipping those that have already
    // been visited at a deeper level
    visit_volume_instances(
        [&](G4VPhysicalVolume const& pv, int depth) {
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

            // Add to name map
            names[pv.GetName()].push_back(&pv);
            // Visit daughters
            return true;
        },
        world);

    return detail::make_label_vector<G4VPhysicalVolume const*>(
        std::move(names),
        [](G4VPhysicalVolume const* pv) { return pv->GetInstanceID(); });
}

//---------------------------------------------------------------------------//
// Global tracking geometry instance: may be nullptr
GeantGeoParams const* g_geant_geo_ = nullptr;

//---------------------------------------------------------------------------//
// Clear the global geometry if it's being destroyed
void destroying_geo(GeantGeoParams const* ggp)
{
    if (ggp == g_geant_geo_)
    {
        g_geant_geo_ = nullptr;
    }
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Set global geometry instance.
 *
 * This allows many parts of the codebase to independently access Geant4
 * metadata. It should be called during initialization of any Celeritas front
 * end that integrates with Geant4. We can't use shared pointers here because
 * of global initialization order issues (the low-level Geant4 objects may be
 * cleared before a static celeritas::GeantGeoParams is destroyed).
 *
 * \note This is not thread safe; it should be done only during setup on the
 * main thread.
 */
void geant_geo(GeantGeoParams const& gp)
{
    CELER_VALIDATE(!g_geant_geo_ || &gp == g_geant_geo_,
                   << "global tracking Geant4 geometry wrapper has already "
                      "been set");
    g_geant_geo_ = &gp;
}

//---------------------------------------------------------------------------//
/*!
 * Access the global geometry instance.
 *
 * This should be used by Geant4 geometry-related helper functions throughout
 * the code base. Make sure to check the result is not null! Do not save a
 * pointer to it either.
 *
 * \return Reference to the global Geant4 wrapper, or nullptr if not set.
 */
GeantGeoParams const* geant_geo()
{
    return g_geant_geo_;
}

//---------------------------------------------------------------------------//
/*!
 * Create from a running Geant4 application.
 */
std::shared_ptr<GeantGeoParams> GeantGeoParams::from_tracking_manager()
{
    auto* world = geant_world_volume();
    CELER_VALIDATE(world,
                   << "cannot create Geant geometry wrapper: Geant4 tracking "
                      "manager is not active");
    return std::make_shared<GeantGeoParams>(world);
}

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
        auto msg = CELER_LOG(debug);
        msg << "GeantGeoParams constructed with a non-navigation world: given "
               "'"
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
        auto* geo_man = G4GeometryManager::GetInstance();
        if (geo_man)
        {
            geo_man->OpenGeometry(this->world());
        }
        else
        {
            CELER_LOG(error) << "Geometry manager was deleted before Geant "
                                "geo had a chance to clean up";
        }
    }
    if (loaded_gdml_)
    {
        reset_geant_geometry();
    }

    // If some part of the code set us up as the global geometry, clear it
    destroying_geo(this);
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
 * (which in Geant4 is not sufficiently unique). See
 * https://jira-geant4.kek.jp/browse/UR-100 and
 * https://github.com/celeritas-project/celeritas/issues/1748 .
 * When this is done, update g4org::PhysicalVolume .
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
    result.replica = this->replica_id(*result.pv);
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
 * Get the volume ID corresponding to a Geant4 physical volume.
 *
 * \note See id_to_geant: the volume instance ID may be non-unique.
 */
VolumeInstanceId
GeantGeoParams::geant_to_id(G4VPhysicalVolume const& volume) const
{
    auto result = id_cast<VolumeInstanceId>(volume.GetInstanceID());
    if (!(result < vol_instances_.size()))
    {
        // Volume is out of range: possibly a PV defined after this geometry
        // class was created??
        result = {};
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Get the replica ID corresponding to a Geant4 physical volume.
 *
 * If the volume is not parameterized or replicated, the result is false.
 */
auto GeantGeoParams::replica_id(G4VPhysicalVolume const& volume) const
    -> ReplicaId
{
    if (volume.VolumeType() == EVolume::kNormal)
        return {};

    auto copy_no = volume.GetCopyNo();
    // NOTE: if this line fails, Geant4 may be returning uninitialized
    // memory on the local thread.
    CELER_ASSERT(copy_no >= 0 && copy_no < volume.GetMultiplicity());
    return id_cast<ReplicaId>(copy_no);
}

//---------------------------------------------------------------------------//
/*!
 * Get the world bbox.
 *
 * This assumes no transformation on the global PV.
 */
BoundingBox<double> GeantGeoParams::get_clhep_bbox() const
{
    auto* world_lv = this->world()->GetLogicalVolume();
    CELER_EXPECT(world_lv);
    G4VSolid const* solid = world_lv->GetSolid();
    CELER_ASSERT(solid);
    G4VisExtent const& extent = solid->GetExtent();

    BoundingBox<double> result{
        {extent.GetXmin(), extent.GetYmin(), extent.GetZmin()},
        {extent.GetXmax(), extent.GetYmax(), extent.GetZmax()}};
    CELER_ENSURE(result);
    return result;
}

//---------------------------------------------------------------------------//
// PRIVATE FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Complete geometry construction.
 */
void GeantGeoParams::build_tracking()
{
    // Close the geometry if needed
    auto* geo_man = G4GeometryManager::GetInstance();
    CELER_ASSERT(geo_man);
    if (!geo_man->IsGeometryClosed())
    {
        CELER_LOG(debug) << "Building geometry manager tracking";
        geo_man->CloseGeometry(
            /* optimize = */ true, /* verbose = */ false, this->world());
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

    // Construct volume labels for physically reachable volumes
    volumes_ = VolumeMap{"volume", make_logical_vol_labels(*this->world())};
    vol_instances_ = VolInstanceMap{"volume instance",
                                    make_physical_vol_labels(*this->world())};
    max_depth_ = get_max_depth(*this->world());

    auto clhep_bbox = this->get_clhep_bbox();
    bbox_ = {convert_from_geant(clhep_bbox.lower().data(), clhep_length),
             convert_from_geant(clhep_bbox.upper().data(), clhep_length)};
    CELER_ENSURE(bbox_);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
