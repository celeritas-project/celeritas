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
#ifdef VECGEOM_USE_SURF
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
#include "corecel/io/Join.hh"
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
#include "geocel/detail/LengthUnits.hh"
#include "geocel/detail/MakeLabelVector.hh"

#include "VecgeomData.hh"  // IWYU pragma: associated
#include "VisitVolumes.hh"

#include "detail/VecgeomCompatibility.hh"
#include "detail/VecgeomSetup.hh"
#include "detail/VecgeomVersion.hh"

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

#ifdef VECGEOM_USE_SURF
#    define VG_SURF_CALL(CODE) CODE
#else
#    define VG_SURF_CALL(CODE) \
        do                     \
        {                      \
        } while (0)
#endif

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
 */
std::vector<Label> make_logical_vol_labels(vecgeom::VPlacedVolume const& world)
{
    using VolT = vecgeom::LogicalVolume;
    std::unordered_map<std::string, std::vector<VolT const*>> names;
    visit_volumes(
        [&](VolT const* lv) {
            CELER_EXPECT(lv);
            std::string name{lv->GetLabel()};
            if (starts_with(name, "[TEMP]"))
            {
                // Temporary volume not directly used in transport, generated
                // by g4vg
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
        },
        &world);

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
    visit_volume_instances(
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
/*!
 * Construct VecGeom objects using the VGDML reader.
 */
auto make_lv_map(std::vector<G4LogicalVolume const*> const& all_lv)
{
    std::unordered_map<G4LogicalVolume const*, ImplVolumeId> result;
    result.reserve(all_lv.size());
    for (auto vol_idx : range(all_lv.size()))
    {
        auto const* lv = all_lv[vol_idx];
        if (lv == nullptr)
        {
            // VecGeom creates fake volumes for boolean volumes
            continue;
        }

        auto&& [iter, inserted]
            = result.insert({lv, id_cast<ImplVolumeId>(vol_idx)});
        if (CELER_UNLIKELY(!inserted))
        {
            // This shouldn't happen...
            CELER_LOG(warning)
                << "Geant4 logical volume " << PrintableLV{iter->first}
                << " maps to multiple VecGeom volume IDs";
        }
    }
    return result;
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
    CELER_VALIDATE(celeritas::geant_geo().expired(),
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
        ScopedProfiling profile_this{"load-vecgeom"};
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
    ScopedProfiling profile_this{"load-vecgeom"};
    ScopedMem record_mem("Converter.convert");
    ScopedTimeLog scoped_time;
    g4vg::Options opts;
    opts.compare_volumes = !celeritas::getenv("G4VG_COMPARE_VOLUMES").empty();
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
#ifdef VECGEOM_USE_SURF
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
 *
 * \todo Instead of VecLv and VecPv, once we we remove `find_volume(G4LV*)`,
 * just pass a vector of volume IDs.
 */
VecgeomParams::VecgeomParams(vecgeom::GeoManager const& geo,
                             Ownership owns,
                             VecLv const& lv,
                             VecPv&& pv)
    : ownership_{owns}
    , g4log_volid_map_{make_lv_map(lv)}
    , g4_pv_map_{std::move(pv)}
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

        if (VecgeomParams::use_surface_tracking())
        {
            this->build_surface_tracking();
        }
        else
        {
            this->build_volume_tracking();
        }

        /*!
         * \todo we still need to make volume tracking information when using
         * CUDA, because we need a GPU world device pointer. We could probably
         * just make a single world physical/logical volume that have the
         * correct IDs.
         */
        if (CELERITAS_USE_CUDA && VecgeomParams::use_surface_tracking())
        {
            this->build_volume_tracking();
        }
    }
    {
        // Save host data
        host_ref_.world_volume = geo.GetWorld();
        host_ref_.max_depth = geo.getMaxDepth();

        if (celeritas::device())
        {
#ifdef VECGEOM_ENABLE_CUDA
            auto& cuda_manager = vecgeom::cxx::CudaManager::Instance();
            device_ref_.world_volume = cuda_manager.world_gpu();
#endif
            device_ref_.max_depth = host_ref_.max_depth;
            CELER_ENSURE(device_ref_.world_volume);
        }
        CELER_ENSURE(host_ref_);
        CELER_ENSURE(!celeritas::device() || device_ref_);
    }
    {
        using namespace vecgeom;
        CELER_ASSERT(host_ref_.world_volume);
        auto const& world = *host_ref_.world_volume;

        // Construct volume labels
        volumes_ = ImplVolumeMap{"volume", make_logical_vol_labels(world)};
        vol_instances_ = VolInstanceMap{"volume instance",
                                        make_physical_vol_labels(world)};

        // Construct ImplVolume -> Volume map
        if (auto geant_geo = celeritas::geant_geo().lock())
        {
            CELER_ASSERT(lv.size() <= this->impl_volumes().size());

            volume_id_map_.resize(this->impl_volumes().size());
            for (auto iv_id : range(id_cast<ImplVolumeId>(lv.size())))
            {
                if (auto* g4lv = lv[iv_id.get()])
                {
                    auto vol_id = geant_geo->geant_to_id(*g4lv);
                    volume_id_map_[iv_id.get()] = vol_id;
                }
            }
        }

        // Save world bbox
        bbox_ = [&world] {
            // Calculate bounding box
            auto bbox_mgr = detail::ABBoxManager_t::Instance();
            Vector3D<real_type> lower, upper;
            bbox_mgr.ComputeABBox(&world, &lower, &upper);
            return BBox{detail::to_array(lower), detail::to_array(upper)};
        }();
    }

    CELER_ENSURE(volumes_);
    CELER_ENSURE(host_ref_);
}

//---------------------------------------------------------------------------//
/*!
 * Clean up vecgeom on destruction.
 */
VecgeomParams::~VecgeomParams()
{
    if (device_ref_)
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

    if (ownership_ == Ownership::value)
    {
        CELER_LOG(debug) << "Clearing VecGeom CPU data";
        vecgeom::GeoManager::Instance().Clear();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Create model parameters corresponding to our internal representation.
 *
 * This could be used to eliminate the "gaps" from the `[TEMP]` volumes.
 */
inp::Model VecgeomParams::make_model_input() const
{
    CELER_LOG(warning) << "VecGeom cannot yet construct model input";
    inp::Model result;

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Get the Geant4 physical volume corresponding to a volume instance ID.
 */
GeantPhysicalInstance VecgeomParams::id_to_geant(VolumeInstanceId id) const
{
    CELER_EXPECT(id < g4_pv_map_.size() || g4_pv_map_.empty());
    if (g4_pv_map_.empty())
    {
        // Model was loaded with VGDML
        return {};
    }

    GeantPhysicalInstance result;
    result.pv = g4_pv_map_[id.unchecked_get()];
    if (result.pv && is_replica(*result.pv))
    {
        // VecGeom volume is a specific instance of a G4PV: get the replica
        // number it corresponds to
        auto& geo_manager = vecgeom::GeoManager::Instance();
#if VECGEOM_VERSION >= 0x020000
        // Constant-time access
        auto* vgpv = geo_manager.GetPlacedVolume(id.get());
#else
        auto* vgpv = geo_manager.FindPlacedVolume(id.get());
#endif
        CELER_ASSERT(vgpv);
        result.replica
            = id_cast<GeantPhysicalInstance::ReplicaId>(vgpv->GetCopyNo());
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Locate the volume ID corresponding to a Geant4 logical volume.
 */
ImplVolumeId VecgeomParams::find_volume(G4LogicalVolume const* volume) const
{
    ImplVolumeId result{};
    if (volume)
    {
        auto iter = g4log_volid_map_.find(volume);
        if (iter != g4log_volid_map_.end())
            result = iter->second;
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * After loading solids, set up VecGeom surface data and copy to GPU.
 */
void VecgeomParams::build_surface_tracking()
{
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
        detail::ABBoxManager_t::Instance().InitABBoxesForCompleteGeometry();
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

        // Check BVH pointers
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
            auto msg = CELER_LOG(error);
            auto msg_pointer = [&msg](auto* p) {
                if (p)
                {
                    msg << p;
                }
                else
                {
                    msg << "nullptr";
                }
            };
            msg << "VecGeom CUDA may not be correctly linked or initialized ("
                   "BVH device pointers are null or inconsistent: ";
            msg_pointer(ptrs.kernel);
            msg << " from Celeritas device kernel, ";
            msg_pointer(ptrs.symbol);
            msg << " from Celeritas runtime symbol, ";
#ifdef VECGEOM_BVHMANAGER_DEVICE
            msg_pointer(bvh_symbol_ptr);
#else
            msg << "unavailable";
#endif
            msg << " from VecGeom runtime symbol)";
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
