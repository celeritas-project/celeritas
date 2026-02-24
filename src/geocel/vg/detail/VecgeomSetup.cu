//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/vg/detail/VecgeomSetup.cu
//---------------------------------------------------------------------------//
#include "VecgeomSetup.hh"

#include <VecGeom/base/Version.h>
#include <VecGeom/management/BVHManager.h>

#if VECGEOM_VERSION >= 0x020000
#    include <VecGeom/management/DeviceGlobals.h>
#endif

#include "corecel/data/DeviceVector.hh"

#if CELERITAS_VECGEOM_SURFACE
#    include <VecGeom/surfaces/cuda/BrepCudaManager.h>
#endif

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/sys/KernelLauncher.device.hh"
#include "corecel/sys/ThreadId.hh"

#if CELERITAS_VECGEOM_SURFACE
using BrepCudaManager = vgbrep::BrepCudaManager<vecgeom::Precision>;
using SurfData = vgbrep::SurfData<vecgeom::Precision>;
#endif

namespace celeritas
{
namespace detail
{
namespace
{
//---------------------------------------------------------------------------//
//! Copy the BVH pointer address to global memory
struct BvhGetter
{
    using pointer_type = CudaBVH_t const*;
    static constexpr char const label[] = "bvh";

    pointer_type* dest{nullptr};

    __device__ void operator()(ThreadId tid)
    {
        CELER_EXPECT(tid == ThreadId{0});
        *dest = vecgeom::cuda::BVHManager::GetBVH(0);
    }
};

//---------------------------------------------------------------------------//
//! Copy the navigation table pointer address to global memory
struct NavIndexGetter
{
    using pointer_type = VgNavIndex const*;
    static constexpr char const label[] = "navindex";

    pointer_type* dest{nullptr};

    __device__ void operator()(ThreadId tid)
    {
        CELER_EXPECT(tid == ThreadId{0});
        *dest = vecgeom::globaldevicegeomdata::gNavIndex;
    }
};

//---------------------------------------------------------------------------//
//! Copy the logical volume pointer table
struct LogicalVolumesGetter
{
    using pointer_type = vecgeom::cuda::LogicalVolume const*;
    static constexpr char const label[] = "logical-volumes";

    pointer_type* dest{nullptr};

    __device__ void operator()(ThreadId tid)
    {
        CELER_EXPECT(tid == ThreadId{0});
#if VECGEOM_VERSION >= 0x020000
        *dest = vecgeom::globaldevicegeomdata::gDeviceLogicalVolumes;
#else
        *dest = nullptr;
#endif
    }
};

//---------------------------------------------------------------------------//
//! Copy the placed volume pointer table
struct PlacedVolumesGetter
{
    using pointer_type = vecgeom::cuda::VPlacedVolume const*;
    static constexpr char const label[] = "placed-volumes";

    pointer_type* dest{nullptr};

    __device__ void operator()(ThreadId tid)
    {
        CELER_EXPECT(tid == ThreadId{0});
#if VECGEOM_VERSION >= 0x020000
        *dest = vecgeom::globaldevicegeomdata::gCompactPlacedVolBuffer;
#else
        *dest = nullptr;
#endif
    }
};

//---------------------------------------------------------------------------//
//! Launch a kernel to copy a value from global memory
template<class GetterT>
auto get_device_pointer()
{
    using pointer_type = typename GetterT::pointer_type;

    // Copy address from inside kernel to GPU global memory
    DeviceVector<pointer_type> temp_global{1, StreamId{}};
    GetterT execute_thread{temp_global.data()};
    static KernelLauncher<decltype(execute_thread)> const launch_kernel(
        "vecgeom-get-" + std::string{GetterT::label});
    launch_kernel(1u, StreamId{}, execute_thread);
    CELER_DEVICE_API_CALL(DeviceSynchronize());

    // Copy address to host
    pointer_type result;
    temp_global.copy_to_host({&result, 1});
    return result;
}

template<class T>
struct InplaceNew
{
    T* ptr;

    __device__ void operator()(ThreadId tid) { new (ptr + tid.get()) T(); }
};

}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Get pointers to the device BVH after setup, for consistency checking.
 */
CudaPointers<CudaBVH_t const> bvh_pointers_device()
{
    CudaPointers<CudaBVH_t const> result;

    // Copy from kernel using 1-thread launch
    result.kernel = get_device_pointer<BvhGetter>();

    // Copy from symbol using runtime API
    CELER_DEVICE_API_CALL(
        MemcpyFromSymbol(static_cast<void*>(&result.symbol),
#if VECGEOM_VERSION >= 0x020000
                         vecgeom::cuda::dBVH<vgbvh_real_type>,
                         sizeof(vecgeom::cuda::dBVH<vgbvh_real_type>),
#else
                         vecgeom::cuda::dBVH,
                         sizeof(vecgeom::cuda::dBVH),
#endif
                         0,
                         CELER_DEVICE_API_SYMBOL(MemcpyDeviceToHost)));
    CELER_DEVICE_API_CALL(DeviceSynchronize());

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Get pointers to the device BVH after setup, for consistency checking.
 */
CudaPointers<VgNavIndex const> navindex_pointers_device()
{
    CudaPointers<VgNavIndex const> result;

    // Copy from kernel using 1-thread launch
    result.kernel = get_device_pointer<NavIndexGetter>();

    // Copy from symbol using runtime API
    CELER_DEVICE_API_CALL(
        MemcpyFromSymbol(static_cast<void*>(&result.symbol),
                         vecgeom::globaldevicegeomdata::gNavIndex,
                         sizeof(vecgeom::globaldevicegeomdata::gNavIndex),
                         0,
                         CELER_DEVICE_API_SYMBOL(MemcpyDeviceToHost)));
    CELER_DEVICE_API_CALL(DeviceSynchronize());

    return result;
}

void check_other_device_pointers()
{
    if constexpr (VECGEOM_VERSION >= 0x020000)
    {
        CELER_VALIDATE(get_device_pointer<LogicalVolumesGetter>() != nullptr,
                       << "failed to copy VG logical volumes to GPU");

        CELER_VALIDATE(get_device_pointer<PlacedVolumesGetter>() != nullptr,
                       << "failed to copy VG placed volumes to GPU");
    }
}

#if CELER_VGNAV == CELER_VGNAV_TUPLE
//---------------------------------------------------------------------------//
/*
 * Default-initialize nav tuple states.
 *
 * This is needed because DeviceVector performs only initialization, not
 * allocation.
 */
void init_navstate_device(Span<VgNavStateImpl> states, StreamId stream)
{
    InplaceNew execute_thread{states.data()};
    static KernelLauncher<decltype(execute_thread)> const launch_kernel(
        "vecgeom-init-navtuple");
    launch_kernel(states.size(), stream, execute_thread);
}
#endif

//---------------------------------------------------------------------------//
// VECGEOM SURFACE
//---------------------------------------------------------------------------//
#if CELERITAS_VECGEOM_SURFACE
void setup_surface_tracking_device(SurfData const& surf_data)
{
    BrepCudaManager::Instance().TransferSurfData(surf_data);
    CELER_DEVICE_API_CALL(DeviceSynchronize());
}

void teardown_surface_tracking_device()
{
    BrepCudaManager::Instance().Cleanup();
}
#endif

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
