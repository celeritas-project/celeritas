//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/GeantGeoParams.cc
//---------------------------------------------------------------------------//
#include "GeantGeoParams.hh"

#include <map>
#include <unordered_map>
#include <vector>
#include <G4GeometryManager.hh>
#include <G4LogicalBorderSurface.hh>
#include <G4LogicalSkinSurface.hh>
#include <G4LogicalVolume.hh>
#include <G4LogicalVolumeStore.hh>
#include <G4Material.hh>
#include <G4PhysicalVolumeStore.hh>
#include <G4Transportation.hh>
#include <G4TransportationManager.hh>
#include <G4VPhysicalVolume.hh>
#include <G4VSolid.hh>
#include <G4Version.hh>
#include <G4VisExtent.hh>

#include "corecel/Assert.hh"
#include "geocel/inp/Model.hh"
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
std::vector<Label> make_logical_vol_labels(G4VPhysicalVolume const& world,
                                           VolumeId::size_type offset)
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
        std::move(names), [offset](G4LogicalVolume const* lv) {
            return lv->GetInstanceID() - offset;
        });
}

//---------------------------------------------------------------------------//
/*!
 * Get a reproducible vector of PV instance ID -> label from the given world.
 */
std::vector<Label> make_physical_vol_labels(G4VPhysicalVolume const& world,
                                            VolumeInstanceId::size_type offset)
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
        std::move(names), [offset](G4VPhysicalVolume const* pv) {
            return pv->GetInstanceID() - offset;
        });
}

//---------------------------------------------------------------------------//
/*!
 * Push back an ordered list of "skin" (boundary) surfaces.
 */
void append_skin_surfaces(GeantGeoParams const& geo,
                          std::vector<G4LogicalSurface const*>& result)
{
    // Translate "skin" (boundary) surfaces
    using G4Surface = G4LogicalSkinSurface;
    std::map<VolumeId, G4Surface const*> temp;
    auto const* table = G4Surface::GetSurfaceTable();
    CELER_ASSERT(table);
#if G4VERSION_NUMBER >= 1120
    for (auto&& [lv, surf] : *table)
#else
    for (auto const* surf : *table)
#endif
    {
#if G4VERSION_NUMBER < 1120
        auto* lv = surf->GetLogicalVolume();
#endif

        CELER_ASSERT(lv);
        auto vol_id = geo.geant_to_id(*lv);
        CELER_ASSERT(vol_id);
        auto iter_inserted = temp.insert({vol_id, surf});
        CELER_ASSERT(iter_inserted.second);
    }

    // Append to table in order
    result.reserve(table->size());
    for (auto const& kv : temp)
    {
        result.push_back(kv.second);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Push back an ordered list of "border" (interface) surfaces.
 */
void append_border_surfaces(GeantGeoParams const& geo,
                            std::vector<G4LogicalSurface const*>& result)
{
    CELER_EXPECT(geo.volume_instances());
    // Translate "border" (interface) surfaces
    using G4Surface = G4LogicalBorderSurface;
    std::map<std::pair<VolumeInstanceId, VolumeInstanceId>, G4Surface const*> temp;
    auto const* table = G4Surface::GetSurfaceTable();
    CELER_ASSERT(table);
#if G4VERSION_NUMBER >= 1070
    for (auto&& [key, surf] : *table)
#else
    for (auto const* surf : *table)
#endif
    {
#if G4VERSION_NUMBER < 1070
        std::pair key{surf->GetVolume1(), surf->GetVolume2()};
#endif
        CELER_ASSERT(key.first);
        auto before = geo.geant_to_id(*key.first);
        CELER_ASSERT(before);
        CELER_ASSERT(key.second);
        auto after = geo.geant_to_id(*key.second);
        CELER_ASSERT(after);
        auto iter_inserted = temp.insert({{before, after}, surf});
        CELER_ASSERT(iter_inserted.second);
    }

    // Add to table in order
    result.reserve(result.size() + table->size());
    for (auto const& kv : temp)
    {
        result.push_back(kv.second);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Get a reproducible list of surfaces.
 */
std::vector<G4LogicalSurface const*> make_surface_vec(GeantGeoParams const& geo)
{
    std::vector<G4LogicalSurface const*> result;
    append_skin_surfaces(geo, result);
    append_border_surfaces(geo, result);

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Construct a volume input from a Geant4 logical volume by mapping IDs.
 */
inp::Volume inp_from_geant(GeantGeoParams const& geo,
                           Label const& label,
                           G4LogicalVolume const& g4lv)
{
    inp::Volume result;
    result.label = label;

    // Set material ID if available
    if (auto* mat = g4lv.GetMaterial())
    {
        result.material = id_cast<GeoMatId>(mat->GetIndex() - geo.mat_offset());
    }
    // Populate volume.children with child volume instances
    auto num_children = g4lv.GetNoDaughters();
    result.children.reserve(num_children);
    for (auto i : range(num_children))
    {
        G4VPhysicalVolume* g4pv = g4lv.GetDaughter(i);
        CELER_ASSERT(g4pv);
        auto vol_inst_id = geo.geant_to_id(*g4pv);
        for (int j = 0, jmax = g4pv->GetMultiplicity(); j < jmax; ++j)
        {
            // TODO: handle replicas correctly by mapping G4PV independently
            // from VIId (each replica gets its own volume instance)
            result.children.push_back(vol_inst_id);
        }
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Create volumes input from Geant4 volumes.
 */
std::vector<inp::Volume> make_inp_volumes(GeantGeoParams const& geo)
{
    std::vector<inp::Volume> result;

    auto const& vol_labels = geo.volumes();
    result.resize(vol_labels.size());

    // Process each logical volume
    for (auto vol_id : range(VolumeId{vol_labels.size()}))
    {
        auto const& label = vol_labels.at(vol_id);
        if (label.empty())
        {
            // This volume isn't part of the world hierarchy
            continue;
        }

        auto* g4lv = geo.id_to_geant(vol_id);
        CELER_ASSERT(g4lv);
        result[vol_id.get()] = inp_from_geant(geo, label, *g4lv);
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Construct a Volume instance input from a Geant4 physical volume.
 */
inp::VolumeInstance inp_from_geant(GeantGeoParams const& geo,
                                   Label const& label,
                                   G4VPhysicalVolume const& g4pv)
{
    inp::VolumeInstance result;
    result.label = label;
    auto* g4lv = g4pv.GetLogicalVolume();
    CELER_ASSERT(g4lv);
    result.volume = geo.geant_to_id(*g4lv);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Create volumes input from Geant4 volumes.
 */
std::vector<inp::VolumeInstance>
make_inp_volume_instances(GeantGeoParams const& geo)
{
    // Process volume instances
    auto const& vol_inst_labels = geo.volume_instances();
    std::vector<inp::VolumeInstance> result(vol_inst_labels.size());

    // Fix copy numbers to avoid invalid read/out-of-bounds
    geo.reset_replica_data();

    for (auto vol_inst_id : range(VolumeInstanceId{vol_inst_labels.size()}))
    {
        auto const& label = vol_inst_labels.at(vol_inst_id);
        if (label.empty())
        {
            // This volume instance isn't part of the world hierarchy
            continue;
        }

        auto g4pv_inst = geo.id_to_geant(vol_inst_id);
        if (!g4pv_inst.pv)
        {
            continue;
        }
        result[vol_inst_id.get()] = inp_from_geant(geo, label, *g4pv_inst.pv);
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Create surfaces input from Geant4 surfaces.
 */
std::vector<inp::Surface> make_inp_surfaces(GeantGeoParams const& geo)
{
    // Process surfaces
    std::vector<inp::Surface> result(geo.num_surfaces());

    for (auto surf_id : range(SurfaceId{geo.num_surfaces()}))
    {
        G4LogicalSurface const* surf_base = geo.id_to_geant(surf_id);
        CELER_ASSERT(surf_base);

        // TODO: deduplicate labels if needed
        result[surf_id.get()].label = surf_base->GetName();

        // Construct surface
        auto& inp_surf = result[surf_id.get()].surface;
        if (auto* surf = dynamic_cast<G4LogicalSkinSurface const*>(surf_base))
        {
            auto* lv = surf->GetLogicalVolume();
            CELER_ASSERT(lv);
            inp_surf = inp::Surface::Boundary{geo.geant_to_id(*lv)};
        }
        else if (auto* surf
                 = dynamic_cast<G4LogicalBorderSurface const*>(surf_base))
        {
            auto* pv_enter = surf->GetVolume1();
            auto* pv_exit = surf->GetVolume2();
            CELER_ASSERT(pv_enter && pv_exit);
            inp_surf = inp::Surface::Interface{geo.geant_to_id(*pv_enter),
                                               geo.geant_to_id(*pv_exit)};
        }
        else
        {
            CELER_ASSERT_UNREACHABLE();
        }
    }
    return result;
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

    data_.world = load_gdml(filename);
    loaded_gdml_ = true;

    this->build_tracking();
    this->build_metadata();

    CELER_ENSURE(volumes_);
    CELER_ENSURE(data_);
}

//---------------------------------------------------------------------------//
/*!
 * Use an existing loaded Geant4 geometry.
 */
GeantGeoParams::GeantGeoParams(G4VPhysicalVolume const* world)
{
    CELER_EXPECT(world);
    data_.world = const_cast<G4VPhysicalVolume*>(world);

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
    CELER_ENSURE(data_);
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
 * Create model params from a Geant4 world volume.
 *
 * \todo Eventually (see #1815) the label map will be stored as part of the
 * volumes and referenced by the geometry, rather than vice versa.
 */
inp::Model GeantGeoParams::make_model_input() const
{
    inp::Model result;

    result.geometry = this->world();
    result.volumes = [this] {
        inp::Volumes result;

        // Get volumes from Geant4 geometry
        result.volumes = make_inp_volumes(*this);
        result.volume_instances = make_inp_volume_instances(*this);

        return result;
    }();
    result.surfaces = [this] {
        inp::Surfaces result;
        result.surfaces = make_inp_surfaces(*this);
        return result;
    }();

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Locate the volume ID corresponding to a Geant4 logical volume.
 */
VolumeId GeantGeoParams::find_volume(G4LogicalVolume const* volume) const
{
    CELER_EXPECT(volume);
    auto result
        = id_cast<VolumeId>(volume->GetInstanceID() - this->lv_offset());
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
    CELER_ASSERT(id < pv_store->size());

    GeantPhysicalInstance result;
    result.pv = (*pv_store)[id.unchecked_get()];
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
    auto index = id.unchecked_get();
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
    auto result = id_cast<VolumeInstanceId>(volume.GetInstanceID()
                                            - this->pv_offset());
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
    // NOTE: if this line fails, you probably need to call \c
    // reset_replica_data from the local thread.
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
/*!
 * Initialize thread-local mutable copy numbers for "replica" volumes.
 *
 * Copy numbers for "replica" volumes (where one instance pretends to be many
 * different volumes) are uninitialized (older Geant4) or initialized to -1.
 * To avoid reading invalid or returning an invalid instance, set all the
 * replica copy numbers to zero.
 */
void GeantGeoParams::reset_replica_data() const
{
    G4PhysicalVolumeStore* pv_store = G4PhysicalVolumeStore::GetInstance();
    CELER_ASSERT(pv_store);
    for (auto* pv : *pv_store)
    {
        if (pv->IsReplicated())
        {
            pv->SetCopyNo(0);
        }
    }
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
    CELER_EXPECT(data_);

    ScopedMem record_mem("GeantGeoParams.build_metadata");

    // Get offset of logical/physical volumes present in unit tests
    data_.lv_offset = [] {
        auto* lv_store = G4LogicalVolumeStore::GetInstance();
        CELER_ASSERT(lv_store && !lv_store->empty());
        return lv_store->front()->GetInstanceID();
    }();
    data_.pv_offset = [] {
        auto* pv_store = G4PhysicalVolumeStore::GetInstance();
        CELER_ASSERT(pv_store && !pv_store->empty());
        return pv_store->front()->GetInstanceID();
    }();
    data_.mat_offset = []() -> GeoMatId::size_type {
        auto* mat_store = G4Material::GetMaterialTable();
        CELER_ASSERT(mat_store);
        if (!mat_store->empty())
        {
            return mat_store->front()->GetIndex();
        }
        return 0;
    }();
    if (this->lv_offset() != 0 || this->pv_offset() != 0
        || this->mat_offset() != 0)
    {
        CELER_LOG(debug) << "Building after volume stores were cleared: "
                         << "lv_offset=" << this->lv_offset()
                         << ", pv_offset=" << this->pv_offset()
                         << ", mat_offset=" << this->mat_offset();
    }

    // Construct volume labels for physically reachable volumes
    volumes_ = VolumeMap{
        "volume", make_logical_vol_labels(*this->world(), this->lv_offset())};
    vol_instances_ = VolInstanceMap{
        "volume instance",
        make_physical_vol_labels(*this->world(), this->pv_offset())};
    surfaces_ = make_surface_vec(*this);
    max_depth_ = get_max_depth(*this->world());

    auto clhep_bbox = this->get_clhep_bbox();
    bbox_ = {convert_from_geant(clhep_bbox.lower().data(), clhep_length),
             convert_from_geant(clhep_bbox.upper().data(), clhep_length)};
    CELER_ENSURE(bbox_);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
