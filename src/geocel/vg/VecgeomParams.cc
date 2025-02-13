//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/vg/VecgeomParams.cc
//---------------------------------------------------------------------------//
#include "VecgeomParams.hh"

#include <cstddef>
#include <vector>
#include <VecGeom/base/Config.h>
#include <VecGeom/base/Cuda.h>
#include <VecGeom/management/ABBoxManager.h>
#include <VecGeom/management/BVHManager.h>
#include <VecGeom/management/GeoManager.h>
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
#include "geocel/GeantGeoUtils.hh"
#include "geocel/detail/LengthUnits.hh"
#include "geocel/detail/MakeLabelVector.hh"

#include "VecgeomData.hh"  // IWYU pragma: associated
#include "VisitVolumes.hh"

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
        [&](VolT const& lv) {
            std::string name{lv.GetLabel()};
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
            names[name].push_back(&lv);
        },
        world);

    return detail::make_label_vector<VolT>(
        std::move(names), [](VolT const& vol) { return vol.id(); });
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
        [&](VolT const& pv, int depth) {
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

            std::string const& name = pv.GetLabel();
            if (starts_with(name, "[TEMP]"))
            {
                // Temporary volume not directly used in tracking
                return false;
            }

            // Add to name map
            names[name].push_back(&pv);
            // Visit daughters
            return true;
        },
        world);

    return detail::make_label_vector<VolT>(
        std::move(names), [](VolT const& vol) { return vol.id(); });
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Whether surface tracking is being used.
 */
bool VecgeomParams::use_surface_tracking()
{
#ifdef VECGEOM_USE_SURF
    return true;
#else
    return false;
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
 * Construct from a GDML input.
 */
VecgeomParams::VecgeomParams(std::string const& filename)
{
    CELER_LOG(status) << "Loading VecGeom geometry from GDML at " << filename;
    if (!ends_with(filename, ".gdml"))
    {
        CELER_LOG(warning) << "Expected '.gdml' extension for GDML input";
    }

    ScopedMem record_mem("VecgeomParams.construct");

    if (VecgeomParams::use_vgdml())
    {
        this->build_volumes_vgdml(filename);
    }
    else
    {
        CELER_NOT_CONFIGURED("VGDML");
    }

    this->build_tracking();
    this->build_data();
    this->build_metadata();

    CELER_ENSURE(volumes_);
    CELER_ENSURE(host_ref_);
}

//---------------------------------------------------------------------------//
/*!
 * Translate a geometry from Geant4.
 */
VecgeomParams::VecgeomParams(G4VPhysicalVolume const* world)
{
    CELER_EXPECT(world);
    ScopedMem record_mem("VecgeomParams.construct");

    this->build_volumes_geant4(world);

    this->build_tracking();
    this->build_data();
    this->build_metadata();

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
        if (VecgeomParams::use_surface_tracking())
        {
            CELER_LOG(debug) << "Clearing VecGeom surface GPU data";
            VG_SURF_CALL(detail::teardown_surface_tracking_device());
        }
        else
        {
            CELER_LOG(debug) << "Clearing VecGeom GPU data";
            VG_CUDA_CALL(vecgeom::CudaManager::Instance().Clear());
        }
    }

    if (VecgeomParams::use_surface_tracking())
    {
        CELER_LOG(debug) << "Clearing SurfModel CPU data";
        VG_SURF_CALL(vgbrep::BrepHelper<real_type>::Instance().ClearData());
    }

    CELER_LOG(debug) << "Clearing VecGeom CPU data";
    vecgeom::GeoManager::Instance().Clear();

    if (loaded_geant4_gdml_)
    {
        reset_geant_geometry();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Get the Geant4 physical volume corresponding to a volume instance ID.
 */
G4VPhysicalVolume const* VecgeomParams::id_to_pv(VolumeInstanceId viid) const
{
    CELER_EXPECT(viid);
    if (viid < g4_pv_map_.size())
    {
        return g4_pv_map_[viid.unchecked_get()];
    }
    return nullptr;
}

//---------------------------------------------------------------------------//
/*!
 * Locate the volume ID corresponding to a Geant4 logical volume.
 */
VolumeId VecgeomParams::find_volume(G4LogicalVolume const* volume) const
{
    VolumeId result{};
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
 * Construct VecGeom objects using the VGDML reader.
 */
void VecgeomParams::build_volumes_vgdml(std::string const& filename)
{
    ScopedProfiling profile_this{"load-vecgeom"};
    ScopedMem record_mem("VecgeomParams.load_geant_geometry");
    ScopedTimeAndRedirect time_and_output_("vgdml::Frontend");

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

//---------------------------------------------------------------------------//
/*!
 * Construct VecGeom objects using Geant4 objects in memory.
 */
void VecgeomParams::build_volumes_geant4(G4VPhysicalVolume const* world)
{
    // Convert the geometry to VecGeom
    ScopedProfiling profile_this{"load-vecgeom"};
    ScopedMem record_mem("Converter.convert");
    ScopedTimeLog scoped_time;
#if CELERITAS_USE_GEANT4
    g4vg::Options opts;
    opts.compare_volumes = !celeritas::getenv("G4VG_COMPARE_VOLUMES").empty();
    opts.scale = static_cast<double>(lengthunits::millimeter);
    opts.append_pointers = false;
    auto result = g4vg::convert(world, opts);
    CELER_ASSERT(result.world != nullptr);
    g4log_volid_map_.reserve(result.logical_volumes.size());
    for (auto vol_idx : range(result.logical_volumes.size()))
    {
        auto const* lv = result.logical_volumes[vol_idx];
        if (lv == nullptr)
        {
            // VecGeom creates fake volumes for boolean volumes
            continue;
        }

        auto&& [iter, inserted]
            = g4log_volid_map_.insert({lv, id_cast<VolumeId>(vol_idx)});
        if (CELER_UNLIKELY(!inserted))
        {
            // This shouldn't happen...
            CELER_LOG(warning)
                << "Geant4 logical volume " << PrintableLV{iter->first}
                << " maps to multiple volume IDs";
        }
    }
    g4_pv_map_ = std::move(result.physical_volumes);

    // Set as world volume
    auto& vg_manager = vecgeom::GeoManager::Instance();
    vg_manager.RegisterPlacedVolume(result.world);
    vg_manager.SetWorldAndClose(result.world);

    // NOTE: setting and closing changes the world
    CELER_ASSERT(vg_manager.GetWorld() != nullptr);
#else
    CELER_DISCARD(world);
    CELER_NOT_CONFIGURED("Geant4");
#endif
}

//---------------------------------------------------------------------------//
/*!
 * After loading solids+volumes, set up VecGeom tracking data and copy to GPU.
 */
void VecgeomParams::build_tracking()
{
    CELER_EXPECT(vecgeom::GeoManager::Instance().GetWorld());
    CELER_LOG(status) << "Initializing tracking information";
    ScopedProfiling profile_this{"initialize-vecgeom"};
    ScopedMem record_mem("VecgeomParams.build_tracking");

    if (VecgeomParams::use_surface_tracking())
    {
        this->build_surface_tracking();
    }
    else
    {
        this->build_volume_tracking();
    }

    /*!
     * \todo we still need to make volume tracking information when using CUDA,
     * because we need a GPU world device pointer. We could probably just make
     * a single world physical/logical volume that have the correct IDs.
     */
    if (CELERITAS_USE_CUDA && VecgeomParams::use_surface_tracking())
    {
        this->build_volume_tracking();
    }
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
        CELER_DEVICE_CHECK_ERROR();
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
        vecgeom::ABBoxManager::Instance().InitABBoxesForCompleteGeometry();
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
            CELER_DEVICE_CALL_PREFIX(DeviceSynchronize());
        }
        {
            CELER_LOG(debug) << "Transferring geometry to GPU";
            ScopedTimeAndRedirect time_and_output_(
                "vecgeom::CudaManager.Synchronize");
            void const* world_top_devptr{nullptr};
            VG_CUDA_CALL(
                world_top_devptr = cuda_manager.Synchronize().GetPtr());
            CELER_DEVICE_CHECK_ERROR();
            CELER_VALIDATE(world_top_devptr != nullptr,
                           << "VecGeom failed to copy geometry to GPU");
        }
        {
            CELER_LOG(debug) << "Initializing BVH on GPU";
            ScopedTimeAndRedirect time_and_output_(
                "vecgeom::BVHManager::DeviceInit");
#if defined(VECGEOM_BVHMANAGER_DEVICE)
            auto* bvh_ptr = BVHManager::DeviceInit();
#elif defined(VECGEOM_ENABLE_CUDA)
            BVHManager::DeviceInit();
#endif
#ifdef VECGEOM_BVHMANAGER_DEVICE
            auto* bvh_symbol_ptr = BVHManager::GetDeviceBVH();
            CELER_VALIDATE(bvh_ptr && bvh_ptr == bvh_symbol_ptr,
                           << "inconsistent BVH device pointer: allocated "
                           << bvh_ptr << " but copy-from-symbol returned "
                           << bvh_symbol_ptr);
#endif
            CELER_DEVICE_CHECK_ERROR();
        }

        // Check BVH pointers
        auto ptrs = detail::bvh_pointers_device();

        vecgeom::cuda::BVH const* bvh_symbol_ptr{nullptr};
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
/*!
 * Construct host/device Celeritas data after setting up VecGeom tracking.
 */
void VecgeomParams::build_data()
{
    ScopedMem record_mem("VecgeomParams.build_data");
    // Save host data
    auto& vg_manager = vecgeom::GeoManager::Instance();
    host_ref_.world_volume = vg_manager.GetWorld();
    host_ref_.max_depth = vg_manager.getMaxDepth();

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

//---------------------------------------------------------------------------//
/*!
 * Construct Celeritas host-only metadata.
 */
void VecgeomParams::build_metadata()
{
    ScopedMem record_mem("VecgeomParams.build_metadata");

    using namespace vecgeom;

    auto& vg_manager = GeoManager::Instance();
    CELER_EXPECT(vg_manager.GetRegisteredVolumesCount() > 0);
    VPlacedVolume const* world = vg_manager.GetWorld();
    CELER_ASSERT(world);

    // Construct volume labels
    volumes_ = VolumeMap{"volume", make_logical_vol_labels(*world)};
    vol_instances_
        = VolInstanceMap{"volume instance", make_physical_vol_labels(*world)};

    // Save world bbox
    bbox_ = [world] {
        // Calculate bounding box
        Vector3D<real_type> lower, upper;
        ABBoxManager::Instance().ComputeABBox(world, &lower, &upper);
        return BBox{detail::to_array(lower), detail::to_array(upper)};
    }();
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
