//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/vg/VecgeomParams.cc
//---------------------------------------------------------------------------//
#include "VecgeomParams.hh"

#include <cstddef>
#include <vector>
#include <VecGeom/base/BVH.h>
#include <VecGeom/base/Config.h>
#include <VecGeom/base/Cuda.h>
#include <VecGeom/base/Version.h>
#include <VecGeom/management/ABBoxManager.h>
#include <VecGeom/management/BVHManager.h>
#include <VecGeom/management/GeoManager.h>
#include <VecGeom/management/ReflFactory.h>
#include <VecGeom/volumes/LogicalVolume.h>
#include <VecGeom/volumes/PlacedVolume.h>

#include "corecel/Config.hh"
#include "corecel/DeviceRuntimeApi.hh"
#ifdef VECGEOM_ENABLE_CUDA
#    include <VecGeom/management/CudaManager.h>
#endif
#if CELERITAS_VECGEOM_SURFACE
#    include <VecGeom/surfaces/BrepHelper.h>
#endif
#ifdef VECGEOM_GDML
#    include <VecGeom/gdml/Frontend.h>
#endif

#if CELERITAS_USE_GEANT4
#    include <G4VG.hh>
#endif

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/cont/Range.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/ScopedTimeAndRedirect.hh"
#include "corecel/io/ScopedTimeLog.hh"
#include "corecel/io/StringUtils.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/Environment.hh"
#include "corecel/sys/ScopedLimitSaver.hh"
#include "corecel/sys/ScopedMem.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "geocel/GeantGeoParams.hh"
#include "geocel/VolumeVisitor.hh"
#include "geocel/detail/LengthUnits.hh"
#include "geocel/detail/MakeLabelVector.hh"

#include "VecgeomData.hh"  // IWYU pragma: associated

#include "detail/VecgeomCompatibility.hh"
#include "detail/VecgeomSetup.hh"

static_assert(std::is_same_v<celeritas::real_type, vecgeom::Precision>,
              "Celeritas and VecGeom real types do not match");

using vecgeom::cxx::BVHManager;

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
// MACROS
// VecGeom interfaces change based on whether CUDA and Surface capabilities
// are available. Use macros to hide the calls.
//---------------------------------------------------------------------------//

#ifdef VECGEOM_ENABLE_CUDA
#    define VG_CUDA_CALL(CODE) CODE
#else
#    define VG_CUDA_CALL(CODE) CELER_UNREACHABLE
#endif

#if CELERITAS_VECGEOM_SURFACE
#    define VG_SURF_CALL(CODE) CODE
#else
#    define VG_SURF_CALL(CODE) \
        do                     \
        {                      \
        } while (0)
#endif

//---------------------------------------------------------------------------//
// HELPER TYPES
//---------------------------------------------------------------------------//

#if VECGEOM_VERSION >= 0x020000
using ABBoxManager_t = vecgeom::ABBoxManager<vecgeom::Precision>;
#else
using ABBoxManager_t = vecgeom::ABBoxManager;
#endif

//---------------------------------------------------------------------------//
// HELPER CLASSES
//---------------------------------------------------------------------------//
class VecgeomVolumeAccessor final
    : public VolumeAccessorInterface<vecgeom::LogicalVolume const*,
                                     vecgeom::VPlacedVolume const*>
{
  public:
    //! Outgoing volume node from an instance
    VolumeRef volume(VolumeInstanceRef parent) final
    {
        CELER_EXPECT(parent);
        auto result = parent->GetLogicalVolume();
        CELER_ENSURE(result);
        return result;
    }

    //! Outgoing edges (placements) from a volume
    ContainerVolInstRef children(VolumeRef parent) final
    {
        CELER_EXPECT(parent);
        auto&& daughters = parent->GetDaughters();
        return ContainerVolInstRef(daughters.begin(), daughters.end());
    }
};

//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Get the verbosity setting for vecgeom.
 */
int vecgeom_verbosity()
{
    static int const result = [] {
        std::string var = celeritas::getenv("VECGEOM_VERBOSE");
        if (var.empty())
        {
            return 0;
        }
        return std::stoi(var);
    }();
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Get a reproducible vector of LV instance ID -> label from the given world.
 *
 * This creates the "implementation" volume map.
 */
std::vector<Label> make_logical_vol_labels(vecgeom::VPlacedVolume const& world)
{
    using VolT = vecgeom::LogicalVolume;
    std::unordered_map<std::string, std::vector<VolT const*>> names;

    VolumeVisitor visit_vol{VecgeomVolumeAccessor{}};
    visit_vol(make_visit_volume_once<vecgeom::LogicalVolume const*>(
                  [&names](VolT const* lv) {
                      CELER_EXPECT(lv);
                      std::string name{lv->GetLabel()};
                      if (starts_with(name, "[TEMP]"))
                      {
                          // Temporary volume not directly used in transport,
                          // generated by g4vg
                          return;
                      }

                      // In case it's loaded by vgdml, strip the pointer
                      if (auto pos = name.find("0x"); pos != std::string::npos)
                      {
                          // Copy pointer as 'extension' and delete from name
                          name.erase(name.begin() + pos, name.end());
                      }

                      // Add to name map
                      names[name].push_back(lv);
                  }),
              world.GetLogicalVolume());

    return detail::make_label_vector<VolT const*>(
        std::move(names), [](VolT const* vol) { return vol->id(); });
}

//---------------------------------------------------------------------------//
/*!
 * Get a reproducible vector of PV instance ID -> label from the given world.
 */
std::vector<Label> make_physical_vol_labels(vecgeom::VPlacedVolume const& world)
{
    using VolT = vecgeom::VPlacedVolume;
    std::unordered_map<VolT const*, int> max_depth;
    std::unordered_map<std::string, std::vector<VolT const*>> names;

    // Visit PVs, mapping names to instances, skipping those that have already
    // been visited at a deeper level
    VolumeVisitor visit_vol{VecgeomVolumeAccessor{}};
    visit_vol(
        [&](VolT const* pv, int depth) {
            auto&& [iter, inserted] = max_depth.insert({pv, depth});
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

            std::string name = pv->GetLabel();
            if (starts_with(name, "[TEMP]"))
            {
                // Temporary volume not directly used in tracking
                return false;
            }

            if (ends_with(name, "_refl")
                && vecgeom::ReflFactory::Instance().IsReflected(
                    pv->GetLogicalVolume()))
            {
                // Strip suffix for consistency with Geant4
                name.erase(name.end() - 5, name.end());
            }

            // Add to name map
            names[std::move(name)].push_back(pv);
            // Visit daughters
            return true;
        },
        &world);

    return detail::make_label_vector<VolT const*>(
        std::move(names), [](VolT const* vol) { return vol->id(); });
}

//---------------------------------------------------------------------------//
vecgeom::VPlacedVolume const&
get_placed_volume(vecgeom::GeoManager const& geo, VgVolumeInstanceId ivi_id)
{
    CELER_EXPECT(ivi_id);

#if VECGEOM_VERSION >= 0x020000
#    define VG_GETPLACEDVOLUME GetPlacedVolume
#else
#    define VG_GETPLACEDVOLUME FindPlacedVolume
#endif
    auto* vgpv = const_cast<vecgeom::GeoManager&>(geo).VG_GETPLACEDVOLUME(
        ivi_id.unchecked_get());
    CELER_ENSURE(vgpv);
    return *vgpv;
}

struct StreamablePointer
{
    void const* ptr{nullptr};
};

std::ostream& operator<<(std::ostream& os, StreamablePointer sp)
{
    if (sp.ptr)
    {
        os << sp.ptr;
    }
    else
    {
        os << "nullptr";
    }

    return os;
}

//---------------------------------------------------------------------------//
/*!
 * Confirm that the BVH device pointers are consistent.
 *
 * RDC linking can cause inline versus noninline methods to return different
 * pointer addresses, leading to bizarre runtime crashes.
 */
void check_bvh_device_pointers()
{
    auto ptrs = detail::bvh_pointers_device();

    detail::CudaBVH_t const* bvh_symbol_ptr{nullptr};
#ifdef VECGEOM_BVHMANAGER_DEVICE
    bvh_symbol_ptr = BVHManager::GetDeviceBVH();
#endif
    if (ptrs.kernel == nullptr || ptrs.kernel != ptrs.symbol
        || (bvh_symbol_ptr && (ptrs.kernel != bvh_symbol_ptr)))
    {
        // It's very bad if the kernel-viewed BVH pointer is null or
        // inconsistent with the VecGeom-provided BVH pointer (only
        // available in very recent VecGeom). It's bad (but not really
        // necessary?) if cudaMemcpyFromSymbol fails when accessed from
        // Celeritas
        CELER_LOG(error)
            << "VecGeom CUDA may not be correctly linked or initialized ("
               "BVH device pointers are null or inconsistent: "
            << StreamablePointer{ptrs.kernel}
            << " from Celeritas device kernel, "
            << StreamablePointer{ptrs.symbol}
            << " from Celeritas runtime symbol, "
#ifdef VECGEOM_BVHMANAGER_DEVICE
            << StreamablePointer{bvh_symbol_ptr}
#else
            << "unavailable"
#endif
            << " from VecGeom runtime symbol)";
    }
}

//---------------------------------------------------------------------------//
/*!
 * Confirm that the navigation index device pointers are consistent.
 */
void check_navindex_device_pointers()
{
    auto ptrs = detail::navindex_pointers_device();
    if (ptrs.kernel == nullptr || ptrs.kernel != ptrs.symbol)
    {
        CELER_LOG(error)
            << "VecGeom CUDA may not be correctly linked or initialized ("
               "navigation index table is null or inconsistent: "
            << StreamablePointer{ptrs.kernel}
            << " from Celeritas device kernel, "
            << StreamablePointer{ptrs.symbol}
            << " from Celeritas runtime symbol)";
    }
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Build by loading a GDML file.
 */
std::shared_ptr<VecgeomParams>
VecgeomParams::from_gdml(std::string const& filename)
{
    if (CELERITAS_USE_GEANT4)
    {
        return VecgeomParams::from_gdml_g4(filename);
    }
    else if (VecgeomParams::use_vgdml())
    {
        return VecgeomParams::from_gdml_vg(filename);
    }
    else
    {
        CELER_NOT_CONFIGURED("Geant4 nor VGDML");
    }
}

//---------------------------------------------------------------------------//
/*!
 * Build by loading a GDML file using Geant4.
 *
 * This mode is incompatible with having an existing run manager. It will clear
 * the geometry once complete.
 */
std::shared_ptr<VecgeomParams>
VecgeomParams::from_gdml_g4(std::string const& filename)
{
    CELER_VALIDATE(celeritas::global_geant_geo().expired(),
                   << "cannot load Geant4 geometry into VecGeom from a "
                      "file name: a global Geant4 geometry already exists");

    // Load temporarily and convert
    return VecgeomParams::from_geant(GeantGeoParams::from_gdml(filename));
}

//---------------------------------------------------------------------------//
/*!
 * Build by loading a GDML file using VecGeom's (buggy) in-house loader.
 */
std::shared_ptr<VecgeomParams>
VecgeomParams::from_gdml_vg(std::string const& filename)
{
    {
        ScopedProfiling profile_this{"vecgeom-vgdml-load"};
        ScopedTimeAndRedirect time_and_output_("vgdml::Frontend");

        CELER_LOG(status) << "Loading VecGeom geometry using VGDML from '"
                          << filename << "'";
#ifdef VECGEOM_GDML
        vgdml::Frontend::Load(
            filename,
            /* validate_xml_schema = */ false,
            /* mm_unit = */ static_cast<double>(lengthunits::millimeter),
            /* verbose = */ vecgeom_verbosity());
#else
        CELER_DISCARD(filename);
        CELER_NOT_CONFIGURED("VGDML");
#endif
    }

    return std::make_shared<VecgeomParams>(
        vecgeom::GeoManager::Instance(), Ownership::value, VecLv{}, VecPv{});
}

//---------------------------------------------------------------------------//
/*!
 * Build from a Geant4 geometry.
 */
std::shared_ptr<VecgeomParams>
VecgeomParams::from_geant(std::shared_ptr<GeantGeoParams const> const& geo)
{
    CELER_EXPECT(geo);
    CELER_LOG(status) << "Loading VecGeom geometry from in-memory Geant4 "
                         "geometry";
#if CELERITAS_USE_GEANT4
    // Convert the geometry to VecGeom
    ScopedProfiling profile_this{"vecgeom-g4vg-load"};
    ScopedMem record_mem("Converter.convert");
    ScopedTimeLog scoped_time;
    g4vg::Options opts;
    opts.compare_volumes = getenv_flag("G4VG_COMPARE_VOLUMES", false).value;
    opts.scale = static_cast<double>(lengthunits::millimeter);
    opts.append_pointers = false;
    opts.verbose = static_cast<bool>(vecgeom_verbosity());
    opts.reflection_factory = false;
    auto result = g4vg::convert(geo->world(), opts);
    CELER_ASSERT(result.world != nullptr);

    // Set as world volume
    // NOTE: setting and closing changes the world
    auto& vg_manager = vecgeom::GeoManager::Instance();
    vg_manager.RegisterPlacedVolume(result.world);
    vg_manager.SetWorldAndClose(result.world);

    return std::make_shared<VecgeomParams>(vg_manager,
                                           Ownership::value,
                                           std::move(result.logical_volumes),
                                           std::move(result.physical_volumes));
#else
    CELER_NOT_CONFIGURED("Geant4");
#endif
}

//---------------------------------------------------------------------------//
/*!
 * Whether surface tracking is being used.
 */
bool VecgeomParams::use_surface_tracking()
{
#if CELERITAS_VECGEOM_SURFACE
    return 1;
#else
    return 0;
#endif
}

//---------------------------------------------------------------------------//
/*!
 * Whether VecGeom GDML is used to load the geometry.
 */
bool VecgeomParams::use_vgdml()
{
#ifdef VECGEOM_GDML
    return true;
#else
    return false;
#endif
}

//---------------------------------------------------------------------------//
/*!
 * Set up vecgeom given existing an already set up VecGeom CPU world.
 */
VecgeomParams::VecgeomParams(vecgeom::GeoManager const& geo,
                             Ownership owns,
                             VecLv const& all_lv,
                             VecPv const& all_pv)
    : host_ownership_{owns}
{
    CELER_VALIDATE(geo.IsClosed(),
                   << "VecGeom geometry was not closed before initialization");
    CELER_VALIDATE(geo.GetWorld() != nullptr,
                   << "VecGeom world was not set before initialization");
    CELER_EXPECT(geo.GetRegisteredVolumesCount() > 0);

    ScopedMem record_mem("VecgeomParams.construct");
    ScopedProfiling profile_this{"initialize-vecgeom"};

    {
        CELER_LOG(status) << "Initializing tracking information";

        if (!VecgeomParams::use_surface_tracking() || CELERITAS_USE_CUDA)
        {
            this->build_volume_tracking();
        }
        if (VecgeomParams::use_surface_tracking())
        {
            this->build_surface_tracking();
        }
    }
    {
        CELER_LOG(status) << "Constructing metadata";

        // Save host data
        HostVal<VecgeomParamsData> host_data;
        host_data.scalars.host_world = geo.GetWorld();
        host_data.scalars.num_volume_levels = geo.getMaxDepth();

        if (celeritas::device())
        {
#ifdef VECGEOM_ENABLE_CUDA
            auto& cuda_manager = vecgeom::cxx::CudaManager::Instance();
            host_data.scalars.device_world = cuda_manager.world_gpu();
#endif
            CELER_ENSURE(host_data.scalars.device_world);
        }

        auto const& world = *geo.GetWorld();

        // Construct volume labels
        impl_volumes_
            = ImplVolumeMap{"impl volume", make_logical_vol_labels(world)};
        impl_vol_instances_ = ImplVolInstanceMap{
            "impl volume instance", make_physical_vol_labels(world)};

        // Resize maps of impl -> canonical IDs
        resize(&host_data.volumes, impl_volumes_.size());
        resize(&host_data.volume_instances, impl_vol_instances_.size());

        geant_geo_ = celeritas::global_geant_geo().lock();

        // Construct Impl vol/inst maps
        if (geant_geo_)
        {
            // Built with Geant4: use G4VG-provided mapping
            for (auto iv_id : range(ImplVolumeId{host_data.volumes.size()}))
            {
                VolumeId vol_id;
                if (iv_id < all_lv.size())
                {
                    if (auto* g4lv = all_lv[iv_id.get()])
                    {
                        vol_id = geant_geo_->geant_to_id(*g4lv);
                    }
                }
                host_data.volumes[iv_id] = vol_id;
            }

            auto& vi_mapper = *geant_geo_->host_ref().vi_mapper;
            for (auto ivi_id :
                 range(ImplVolInstanceId{host_data.volume_instances.size()}))
            {
                VolumeInstanceId vol_inst_id;
                if (ivi_id < all_pv.size())
                {
                    if (auto* g4pv = all_pv[ivi_id.get()])
                    {
                        // To support replica/param volumes, use the copy
                        // number from the corresponding Vecgeom placed volume
                        vol_inst_id = vi_mapper.geant_to_id(
                            *g4pv, get_placed_volume(geo, ivi_id).GetCopyNo());
                    }
                }
                host_data.volume_instances[ivi_id] = vol_inst_id;
            }
        }
        else
        {
            // Built with VGDML: create one-to-one mapping
            for (auto iv_id : range(ImplVolumeId{host_data.volumes.size()}))
            {
                host_data.volumes[iv_id] = id_cast<VolumeId>(iv_id.get());
            }
            for (auto ivi_id :
                 range(ImplVolInstanceId{host_data.volume_instances.size()}))
            {
                host_data.volume_instances[ivi_id]
                    = id_cast<VolumeInstanceId>(ivi_id.get());
            }
        }
        CELER_ASSERT(host_data);
        data_ = CollectionMirror{std::move(host_data)};

        // Save world bbox
        bbox_ = [&world] {
            // Calculate bounding box
            auto bbox_mgr = ABBoxManager_t::Instance();
            VgReal3 lower, upper;
            bbox_mgr.ComputeABBox(&world, &lower, &upper);
            return BBox{detail::to_array(lower), detail::to_array(upper)};
        }();
    }

    CELER_ENSURE(impl_volumes_);
    CELER_ENSURE(data_);
}

//---------------------------------------------------------------------------//
/*!
 * Clean up vecgeom on destruction.
 */
VecgeomParams::~VecgeomParams()
{
    if (device_ownership_ == Ownership::value)
    {
        CELER_LOG(debug)
            << "Clearing VecGeom "
            << (VecgeomParams::use_surface_tracking() ? "surface" : "volume")
            << "GPU data";
        try
        {
            if (VecgeomParams::use_surface_tracking())
            {
                VG_SURF_CALL(detail::teardown_surface_tracking_device());
            }
            else
            {
                VG_CUDA_CALL(vecgeom::CudaManager::Instance().Clear());
            }
        }
        catch (std::exception const& e)
        {
            CELER_LOG(critical)
                << "Failed during VecGeom device cleanup: " << e.what();
        }
    }

    if (host_ownership_ == Ownership::value)
    {
        if (VecgeomParams::use_surface_tracking())
        {
            CELER_LOG(debug) << "Clearing SurfModel CPU data";
        }
        try
        {
            VG_SURF_CALL(vgbrep::BrepHelper<real_type>::Instance().ClearData());
        }
        catch (std::exception const& e)
        {
            CELER_LOG(critical)
                << "Failed during VecGeom surface model cleanup: " << e.what();
        }
        CELER_LOG(debug) << "Clearing VecGeom CPU data";
        vecgeom::GeoManager::Instance().Clear();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Create model parameters corresponding to our internal representation.
 *
 * Currently this creates a one-to-one mapping for use when constructed from
 * VGDML rather than Geant4.
 */
inp::Model VecgeomParams::make_model_input() const
{
    CELER_LOG(warning)
        << R"(VecGeom standalone model input is not fully implemented)";

    inp::Model result;
    inp::Volumes& v = result.volumes;
    v.volumes.resize(impl_volumes_.size());
    v.volume_instances.resize(impl_vol_instances_.size());

    // Create one-to-one map for logical volumes
    for (auto iv_id : range(ImplVolumeId{impl_volumes_.size()}))
    {
        auto const& label = impl_volumes_.at(iv_id);
        if (label.name.empty())
        {
            continue;
        }

        v.volumes[iv_id.get()].label = label;
        v.volumes[iv_id.get()].material = GeoMatId{0};
    }

    // Create one-to-one map for placed volumes
    auto const& geo = vecgeom::GeoManager::Instance();
    for (auto ivi_id : range(ImplVolInstanceId{impl_vol_instances_.size()}))
    {
        auto const& label = impl_vol_instances_.at(ivi_id);
        if (label.name.empty())
        {
            continue;
        }

        auto const& placed_vol = get_placed_volume(geo, ivi_id);

        v.volume_instances[ivi_id.get()].label = label;
        // Save the underlying volume for this instance
        v.volume_instances[ivi_id.get()].volume
            = id_cast<VolumeId>(placed_vol.GetLogicalVolume()->id());
    }

    result.volumes.world
        = id_cast<VolumeId>(geo.GetWorld()->GetLogicalVolume()->id());
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * After loading solids, set up VecGeom surface data and copy to GPU.
 */
void VecgeomParams::build_surface_tracking()
{
    static int initialization_count{0};
    CELER_VALIDATE(initialization_count == 0,
                   << "VecGeom surface geometry currently crashes if "
                      "recreated during an execution (you may call BeamOn "
                      "only once)");

    VG_SURF_CALL(auto& brep_helper = vgbrep::BrepHelper<real_type>::Instance());
    VG_SURF_CALL(brep_helper.SetVerbosity(vecgeom_verbosity()));

    {
        CELER_LOG(debug) << "Creating surfaces";
        ScopedTimeAndRedirect time_and_output_("BrepHelper::Convert");
        VG_SURF_CALL(CELER_VALIDATE(brep_helper.Convert(),
                                    << "failed to convert VecGeom to "
                                       "surfaces"));
        if (vecgeom_verbosity() > 1)
        {
            VG_SURF_CALL(brep_helper.PrintSurfData());
        }
        // Prevent us from accidentally rebuilding and getting a segfault
        // See https://its.cern.ch/jira/projects/VECGEOM/issues/VECGEOM-634
        ++initialization_count;
    }

    if (celeritas::device())
    {
        CELER_LOG(debug) << "Transferring surface data to GPU";
        ScopedTimeAndRedirect time_and_output_(
            "BrepCudaManager::TransferSurfData");

        VG_SURF_CALL(
            detail::setup_surface_tracking_device(brep_helper.GetSurfData()));
        CELER_DEVICE_API_CALL(PeekAtLastError());
    }
}

//---------------------------------------------------------------------------//
/*!
 * After loading solids, set up VecGeom tracking data and copy to GPU.
 *
 * After instantiating the CUDA manager, which changes the stack limits, we
 * adjust the stack size based on a user variable due to VecGeom recursive
 * virtual function calls. This is necessary for deeply nested geometry such as
 * CMS, as well as certain cases with debug symbols and assertions.
 *
 * See https://github.com/celeritas-project/celeritas/issues/614
 */
void VecgeomParams::build_volume_tracking()
{
    CELER_EXPECT(vecgeom::GeoManager::Instance().GetWorld());

    {
        ScopedTimeAndRedirect time_and_output_("vecgeom::ABBoxManager");
        ABBoxManager_t::Instance().InitABBoxesForCompleteGeometry();
    }

    // Init the bounding volume hierarchy structure
    BVHManager::Init();

    if (celeritas::device())
    {
        {
            // NOTE: this *MUST* be the first time the CUDA manager is called,
            // otherwise we can't restore limits.
            ScopedLimitSaver save_cuda_limits;
            VG_CUDA_CALL(vecgeom::cxx::CudaManager::Instance());
        }

        // Set custom stack and heap size now that it's been initialized
        if (std::string var = celeritas::getenv("CUDA_STACK_SIZE");
            !var.empty())
        {
            int stack_size = std::stoi(var);
            CELER_VALIDATE(stack_size > 0,
                           << "invalid CUDA_STACK_SIZE=" << stack_size
                           << " (must be positive)");
            set_cuda_stack_size(stack_size);
        }
        else if constexpr (CELERITAS_DEBUG)
        {
            // Default to a large stack size due to debugging code.
            set_cuda_stack_size(16384);
        }

        if (std::string var = celeritas::getenv("CUDA_HEAP_SIZE"); !var.empty())
        {
            int heap_size = std::stoi(var);
            CELER_VALIDATE(heap_size > 0,
                           << "invalid CUDA_HEAP_SIZE=" << heap_size
                           << " (must be positive)");
            set_cuda_heap_size(heap_size);
        }

#ifdef VECGEOM_ENABLE_CUDA
        auto& cuda_manager = vecgeom::cxx::CudaManager::Instance();
        cuda_manager.set_verbose(vecgeom_verbosity());
#endif
        {
            CELER_LOG(debug) << "Converting to CUDA geometry";
            ScopedTimeAndRedirect time_and_output_(
                "vecgeom::CudaManager.LoadGeometry");

            VG_CUDA_CALL(cuda_manager.LoadGeometry());
            CELER_DEVICE_API_CALL(DeviceSynchronize());
        }
        {
            CELER_LOG(debug) << "Transferring geometry to GPU";
            ScopedTimeAndRedirect time_and_output_(
                "vecgeom::CudaManager.Synchronize");
            void const* world_top_devptr{nullptr};
            VG_CUDA_CALL(
                world_top_devptr = cuda_manager.Synchronize().GetPtr());
            CELER_DEVICE_API_CALL(PeekAtLastError());
            CELER_VALIDATE(world_top_devptr != nullptr,
                           << "VecGeom failed to copy geometry to GPU");
        }
        {
            CELER_LOG(debug) << "Initializing BVH on GPU";
            ScopedTimeAndRedirect time_and_output_(
                "vecgeom::BVHManager::DeviceInit");
#if defined(VECGEOM_BVHMANAGER_DEVICE)
            auto* bvh_ptr = BVHManager::DeviceInit();
            auto* bvh_symbol_ptr = BVHManager::GetDeviceBVH();
            CELER_VALIDATE(bvh_ptr && bvh_ptr == bvh_symbol_ptr,
                           << "inconsistent BVH device pointer: allocated "
                           << bvh_ptr << " but copy-from-symbol returned "
                           << bvh_symbol_ptr);
#elif defined(VECGEOM_ENABLE_CUDA)
            BVHManager::DeviceInit();
#endif
            CELER_DEVICE_API_CALL(PeekAtLastError());
        }

        check_bvh_device_pointers();
        check_navindex_device_pointers();

        device_ownership_ = Ownership::value;
    }
}

//---------------------------------------------------------------------------//
// EXPLICIT TEMPLATE INSTANTIATION
//---------------------------------------------------------------------------//

template class CollectionMirror<VecgeomParamsData>;
template class ParamsDataInterface<VecgeomParamsData>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
