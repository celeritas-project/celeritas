//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/GeantGeoParams.cc
//---------------------------------------------------------------------------//
#include "GeantGeoParams.hh"

#include <map>
#include <unordered_map>
#include <unordered_set>
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
#include <G4VSensitiveDetector.hh>
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
#include "corecel/sys/ScopedProfiling.hh"

#include "GeantGdmlLoader.hh"
#include "GeantGeoUtils.hh"
#include "GeantUtils.hh"
#include "ScopedGeantExceptionHandler.hh"
#include "ScopedGeantLogger.hh"
#include "g4/Convert.hh"  // IWYU pragma: associated
#include "g4/GeantGeoData.hh"  // IWYU pragma: associated

#include "detail/MakeLabelVector.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Get a reproducible vector of LV instance ID -> label from the given world.
 */
std::vector<Label>
make_logical_vol_labels(detail::GeantVolumeInstanceMapper const& vi_mapper,
                        ImplVolumeId::size_type lv_offset)
{
    std::unordered_set<G4LogicalVolume const*> visited_lv;
    std::unordered_map<std::string, std::vector<G4LogicalVolume const*>> names;

    for (auto vi_id : range(VolumeInstanceId{vi_mapper.size()}))
    {
        G4LogicalVolume const* lv = &vi_mapper.logical_volume(vi_id);
        if (!visited_lv.insert(lv).second)
        {
            // LV already has been included
            continue;
        }
        std::string name = lv->GetName();
        if (name.empty())
        {
            CELER_LOG(debug)
                << "Empty name for reachable LV id=" << lv->GetInstanceID();
            name = "[UNTITLED]";
        }
        // Add to name map
        names[std::move(name)].push_back(lv);
    }

    return detail::make_label_vector<G4LogicalVolume const*>(
        std::move(names), [lv_offset](G4LogicalVolume const* lv) {
            return lv->GetInstanceID() - lv_offset;
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
    size_type num_null_surfaces{0};
#if G4VERSION_NUMBER >= 1120
    for (auto&& [lv, surf] : *table)
#else
    for (auto const* surf : *table)
#endif
    {
        if (!surf)
        {
            ++num_null_surfaces;
            continue;
        }

#if G4VERSION_NUMBER < 1120
        auto* lv = surf->GetLogicalVolume();
#endif
        if (!lv)
        {
            CELER_LOG(warning) << "G4 skin surface '" << surf->GetName()
                               << "' references a null logical volume";
            continue;
        }

        auto vol_id = geo.geant_to_id(*lv);
        CELER_ASSERT(vol_id);
        auto iter_inserted = temp.insert({vol_id, surf});
        CELER_ASSERT(iter_inserted.second);
    }

    if (num_null_surfaces != 0)
    {
        CELER_LOG(warning) << "Geant4 skin surface table contains "
                           << num_null_surfaces << " null surface"
                           << (num_null_surfaces > 1 ? "s" : "");
    }

    // Append to table in order
    result.reserve(result.size() + temp.size());
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
    // Translate "border" (interface) surfaces
    using G4Surface = G4LogicalBorderSurface;
    std::map<std::pair<VolumeInstanceId, VolumeInstanceId>, G4Surface const*> temp;
    auto const* table = G4Surface::GetSurfaceTable();
    CELER_ASSERT(table);

    size_type num_null_surfaces{0};
#if G4VERSION_NUMBER >= 1070
    for (auto&& [key, surf] : *table)
#else
    for (auto const* surf : *table)
#endif
    {
        if (!surf)
        {
            ++num_null_surfaces;
            continue;
        }

#if G4VERSION_NUMBER < 1070
        std::pair key{surf->GetVolume1(), surf->GetVolume2()};
#endif
        if (!(key.first && key.second))
        {
            CELER_LOG(warning) << "G4 border surface '" << surf->GetName()
                               << "' references a null physical volume";
            continue;
        }
        if (key.first->IsReplicated() || key.second->IsReplicated())
        {
            CELER_LOG(error) << "G4 border surface '" << surf->GetName()
                             << "' uses replica/parameterised volumes: these "
                                "will be ignored!";
            continue;
        }
        auto before = geo.geant_to_id(*key.first);
        CELER_ASSERT(before);
        auto after = geo.geant_to_id(*key.second);
        CELER_ASSERT(after);
        auto iter_inserted = temp.insert({{before, after}, surf});
        CELER_ASSERT(iter_inserted.second);
    }

    if (num_null_surfaces != 0)
    {
        CELER_LOG(warning) << "Geant4 border surface table contains "
                           << num_null_surfaces << " null surface"
                           << (num_null_surfaces > 1 ? "s" : "");
    }

    // Add to table in order
    result.reserve(result.size() + temp.size());
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
 * Create volumes input from Geant4 volumes.
 *
 * Logical volume labels have already been "uniquified" as part of the
 * implementation volume ID.
 *
 * \todo This will change slightly if we remap logical volumes to be ordered
 * and filtered. For now there's a direct correspondence between implementation
 * volume and canonical volume. We probably also want to add uniquifying
 * extensions and to map `_refl` to the same name (but different extension) to
 * correctly associate detectors.
 */
std::vector<inp::Volume> make_inp_volumes(GeantGeoParams const& geo)
{
    std::vector<inp::Volume> result;

    auto const& vol_labels = geo.impl_volumes();
    result.resize(vol_labels.size());

    // Process each logical volume
    for (auto iv_id : range(ImplVolumeId{vol_labels.size()}))
    {
        auto const& label = vol_labels.at(iv_id);
        if (label.empty())
        {
            // This volume isn't part of the world hierarchy
            continue;
        }

        auto vol_id = geo.volume_id(iv_id);
        auto& g4lv = *geo.id_to_geant(vol_id);

        // Set the label and material
        auto& vol_inp = result[vol_id.get()];
        vol_inp.label = label;
        vol_inp.material = [&geo, mat = g4lv.GetMaterial()]() -> GeoMatId {
            if (!mat)
                return {};
            return geo.geant_to_id(*mat);
        }();

        // Populate volume.children with child volume instance IDs
        auto num_children = g4lv.GetNoDaughters();
        vol_inp.children.reserve(num_children);
        for (auto i : range(num_children))
        {
            // One physical volume can correspond to multiple volume instances
            // if using replica/parameterized volumes
            G4VPhysicalVolume* g4pv = g4lv.GetDaughter(i);
            CELER_ASSERT(g4pv);
            for (int j = 0, jmax = g4pv->GetMultiplicity(); j < jmax; ++j)
            {
                if (g4pv->IsReplicated())
                {
                    // Note that the copy number is thread-local and
                    // "ephemeral": there should be no side effects.
                    // See also GeantVolumeInstanceMapper::update_replica .
                    g4pv->SetCopyNo(j);
                }
                auto vol_inst_id = geo.geant_to_id(*g4pv);
                vol_inp.children.push_back(vol_inst_id);
            }
        }
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Create volume instance input data.
 */
std::vector<inp::VolumeInstance>
make_inp_volume_instances(GeantGeoParams const& geo)
{
    CELER_ASSERT(geo.host_ref().vi_mapper);
    auto const& vi_mapper = *geo.host_ref().vi_mapper;

    std::vector<inp::VolumeInstance> result(vi_mapper.size());
    std::unordered_map<std::string, size_type> name_count;

    for (auto vi_idx : range(vi_mapper.size()))
    {
        auto const& g4pv = vi_mapper.id_to_geant(VolumeInstanceId{vi_idx});
        auto& vi_inp = result[vi_idx];

        // Construct label and unique extension
        auto const& name = g4pv.GetName();
        size_type count = name_count[name]++;
        vi_inp.label = {name, std::to_string(count)};

        // Map the corresponding VolumeId
        auto* g4lv = g4pv.GetLogicalVolume();
        CELER_ASSERT(g4lv);
        vi_inp.volume = geo.geant_to_id(*g4lv);
        if (!vi_inp.volume)
        {
            CELER_LOG(error) << "No canonical volume ID corresponds to "
                             << StreamableLV{g4lv} << " in physical volume "
                             << vi_inp.label;
            vi_inp.label = {};
            continue;
        }
    }

    // Remove extensions if only one volume with that name was present
    for (auto& vi_inp : result)
    {
        if (name_count.at(vi_inp.label.name) == 1)
        {
            vi_inp.label.ext = {};
        }
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

        // TODO: deduplicate labels
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
/*!
 * Create sensitive detectors input from Geant4 sensitive detectors.
 */
std::vector<inp::Detector> make_inp_detectors(GeantGeoParams const& geo)
{
    // Process detectors
    std::vector<inp::Detector> result;

    auto const& vol_labels = geo.impl_volumes();

    std::unordered_map<G4VSensitiveDetector const*, std::vector<VolumeId>>
        detector_map;

    // Process each logical volume
    for (auto iv_id : range(ImplVolumeId{vol_labels.size()}))
    {
        auto vol_id = geo.volume_id(iv_id);
        if (!vol_id)
        {
            // This volume isn't part of the world hierarchy
            continue;
        }
        auto& g4lv = *geo.id_to_geant(vol_id);

        // Add volume id to detector map if it is in a detector region
        if (G4VSensitiveDetector const* sd = g4lv.GetSensitiveDetector())
        {
            detector_map[sd].push_back(vol_id);
        }
    }

    // Loop over detector_map and add detectors to result vector
    for (auto&& [sd, volumes] : detector_map)
    {
        inp::Detector detector;
        detector.label = sd->GetName();
        detector.volumes = std::move(volumes);
        std::sort(detector.volumes.begin(), detector.volumes.end());
        result.push_back(detector);
    }

    std::sort(result.begin(), result.end(), [](auto& left, auto& right) {
        return left.volumes.front() < right.volumes.front();
    });

    return result;
}

//---------------------------------------------------------------------------//
//! Global tracking geometry instance: may be nullptr
// Note that this is safe to declare statically: see
// https://en.cppreference.com/w/cpp/memory/weak_ptr/weak_ptr
std::weak_ptr<GeantGeoParams const> g_geant_geo_;

//---------------------------------------------------------------------------//
//! Placeholder SD class for generating model data from GDML
class GdmlSensitiveDetector final : public G4VSensitiveDetector
{
  public:
    GdmlSensitiveDetector(std::string const& name) : G4VSensitiveDetector{name}
    {
    }

    void Initialize(G4HCofThisEvent*) final {}
    bool ProcessHits(G4Step*, G4TouchableHistory*) final { return false; }
};

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
 * \note This should be done only during setup on the main thread.
 */
void global_geant_geo(std::shared_ptr<GeantGeoParams const> const& gp)
{
    CELER_LOG(debug) << (gp ? "Setting" : "Clearing")
                     << " celeritas::global_geant_geo";
    CELER_VALIDATE(
        g_geant_geo_.expired() ||
            [&] {
                auto old_p = g_geant_geo_.lock();
                return !old_p || old_p == gp;
            }(),
        << "global tracking Geant4 geometry wrapper has already been set");
    g_geant_geo_ = gp;
}

//---------------------------------------------------------------------------//
/*!
 * Access the global geometry instance.
 *
 * This can be used by Geant4 geometry-related helper functions throughout
 * the code base.
 *
 * \return Weak pointer to the global Geant4 wrapper, which may be null.
 */
std::weak_ptr<GeantGeoParams const> const& global_geant_geo()
{
    return g_geant_geo_;
}

//---------------------------------------------------------------------------//
/*!
 * Create from a running Geant4 application.
 *
 * It saves the result to the global Celeritas Geant4 geometry weak pointer \c
 * global_geant_geo.
 */
std::shared_ptr<GeantGeoParams> GeantGeoParams::from_tracking_manager()
{
    auto* world = geant_world_volume();
    CELER_VALIDATE(world,
                   << "cannot create Geant geometry wrapper: Geant4 tracking "
                      "manager is not active");
    auto result = std::make_shared<GeantGeoParams>(world, Ownership::reference);
    celeritas::global_geant_geo(result);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Construct from a GDML input.
 *
 * This assumes that Celeritas is driving and will manage Geant4 logging
 * and exceptions. It saves the result to the global Celeritas Geant4 geometry
 * weak pointer \c global_geant_geo.
 *
 * Due to limitations in the Geant4 GDML code, this task \c must be performed
 * from the main thread.
 *
 * It also loads sensitive detectors and assigns dummy sensitive detectors to
 * volumes annotated with <code>auxiliary auxtype="SensDet"</code> tags. It
 * creates one detector per unique \c auxvalue name and shares that one among
 * the volumes that use the same detector name. The resulting \c GeantGeoParams
 * class retains ownership of the created detectors. Since this function is
 * only called on the main thread, and the \c SensitiveDetector getter/setter
 * on \c G4LogicalVolume uses a thread-local "split" class, <em>worker threads
 * will not see the sensitive detectors this loader creates</em>. Use \c
 * celeritas::DetectorConstruction if thread-local detectors are needed.
 */
std::shared_ptr<GeantGeoParams>
GeantGeoParams::from_gdml(std::string const& filename)
{
    ScopedMem record_mem("GeantGeoParams.construct");

    ScopedGeantLogger logger(celeritas::world_logger());
    ScopedGeantExceptionHandler exception_handler;

    disable_geant_signal_handler();

    if (!ends_with(filename, ".gdml"))
    {
        CELER_LOG(warning) << "Expected '.gdml' extension for GDML input";
    }

    // Load world and detectors
    auto loaded = [&filename] {
        GeantGdmlLoader::Options opts;
        opts.detectors = true;
        GeantGdmlLoader load(opts);
        return load(filename);
    }();

    // Build placeholder SD
    using MapDetCIter = GeantGdmlLoader::MapDetectors::const_iterator;
    GeantGeoParams::MapStrDetector built_detectors;
    foreach_detector(
        loaded.detectors,
        [&built_detectors](MapDetCIter iter, MapDetCIter stop) {
            // Construct an SD based on the name
            auto sd = std::make_shared<GdmlSensitiveDetector>(iter->first);
            built_detectors.emplace(iter->first, sd);

            // Attach sensitive detectors
            for (; iter != stop; ++iter)
            {
                CELER_LOG(debug)
                    << "Attaching dummy GDML SD '" << sd->GetName()
                    << "' to volume '" << iter->second->GetName() << "'";
                iter->second->SetSensitiveDetector(sd.get());
            }
        });

    // Create geo params
    auto result
        = std::make_shared<GeantGeoParams>(loaded.world, Ownership::value);
    // Set detectors (hack)
    result->built_detectors_ = std::move(built_detectors);

    // Save for use outside in Celeritas
    celeritas::global_geant_geo(result);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Use an existing loaded Geant4 geometry.
 */
GeantGeoParams::GeantGeoParams(G4VPhysicalVolume const* world, Ownership owns)
    : ownership_{owns}
{
    CELER_EXPECT(world);
    data_.world = const_cast<G4VPhysicalVolume*>(world);

    ScopedMem record_mem("GeantGeoParams.construct");
    ScopedProfiling profile_this{"geant-geo-construct"};

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

    this->build_metadata();

    CELER_ENSURE(impl_volumes_);
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
    if (ownership_ == Ownership::value)
    {
        reset_geant_geometry();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Create model params from a Geant4 world volume.
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
        result.world = this->geant_to_id(*(this->world()->GetLogicalVolume()));

        return result;
    }();
    result.surfaces = [this] {
        inp::Surfaces result;
        result.surfaces = make_inp_surfaces(*this);
        return result;
    }();

    result.detectors = [this] {
        inp::Detectors result;
        result.detectors = make_inp_detectors(*this);
        return result;
    }();

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Locate the volume ID corresponding to a Geant4 logical volume.
 */
VolumeId GeantGeoParams::geant_to_id(G4LogicalVolume const& volume) const
{
    auto result
        = id_cast<ImplVolumeId>(volume.GetInstanceID() - this->lv_offset());
    if (!(result < impl_volumes_.size()))
    {
        // Volume is out of range: possibly an LV defined after this geometry
        // class was created
        result = {};
    }
    return this->volume_id(result);
}

//---------------------------------------------------------------------------//
/*!
 * Get the Geant4 logical volume corresponding to a volume ID.
 *
 * If the input volume ID is unassigned, a null pointer will be returned.
 */
G4LogicalVolume const* GeantGeoParams::id_to_geant(VolumeId id) const
{
    CELER_EXPECT(!id || id < impl_volumes_.size());
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
 * Get the geometry material ID for a logical volume.
 */
GeoMatId GeantGeoParams::geant_to_id(G4Material const& g4mat) const
{
    return id_cast<GeoMatId>(g4mat.GetIndex() - this->mat_offset());
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
 * Construct Celeritas host-only metadata.
 */
void GeantGeoParams::build_metadata()
{
    CELER_EXPECT(data_.world);
    ScopedMem record_mem("GeantGeoParams.build_metadata");

    // Get offsets used to map material and impl volume IDs
    data_.lv_offset = [] {
        auto* lv_store = G4LogicalVolumeStore::GetInstance();
        CELER_ASSERT(lv_store && !lv_store->empty());
        return lv_store->front()->GetInstanceID();
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
    if (this->lv_offset() != 0 || this->mat_offset() != 0)
    {
        CELER_LOG(debug) << "Building after volume stores were cleared: "
                         << "lv_offset=" << this->lv_offset()
                         << ", mat_offset=" << this->mat_offset();
    }

    // Construct volume instance mapper
    vi_mapper_ = detail::GeantVolumeInstanceMapper(*this->world());
    data_.vi_mapper = &vi_mapper_;

    // Construct volume labels for physically reachable volumes
    impl_volumes_ = ImplVolumeMap{
        "impl volume", make_logical_vol_labels(vi_mapper_, this->lv_offset())};
    surfaces_ = make_surface_vec(*this);

    auto clhep_bbox = this->get_clhep_bbox();
    bbox_ = {convert_from_geant(clhep_bbox.lower().data(), clhep_length),
             convert_from_geant(clhep_bbox.upper().data(), clhep_length)};
    CELER_ENSURE(bbox_);
    CELER_ENSURE(data_);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
