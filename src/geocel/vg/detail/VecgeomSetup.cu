//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/vg/detail/VecgeomSetup.cu
//---------------------------------------------------------------------------//
#include "VecgeomSetup.hh"

#include <VecGeom/management/BVHManager.h>

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

    CELER_FUNCTION void operator()(ThreadId tid)
    {
        CELER_EXPECT(tid == ThreadId{0});
        *dest = vecgeom::cuda::BVHManager::GetBVH(0);
    }
};

//---------------------------------------------------------------------------//
//! Copy the navigation table pointer address to global memory
struct NavIndexGetter
{
    using pointer_type = NavIndex_t const*;
    static constexpr char const label[] = "navindex";

    pointer_type* dest{nullptr};

    __device__ void operator()(ThreadId tid)
    {
        CELER_EXPECT(tid == ThreadId{0});
        *dest = vecgeom::globaldevicegeomdata::gNavIndex;
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
        MemcpyFromSymbol(&result.symbol,
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
CudaPointers<NavIndex_t const> navindex_pointers_device()
{
    CudaPointers<NavIndex_t const> result;

    // Copy from kernel using 1-thread launch
    result.kernel = get_device_pointer<NavIndexGetter>();

    // Copy from symbol using runtime API
    CELER_DEVICE_API_CALL(
        MemcpyFromSymbol(&result.symbol,
                         vecgeom::globaldevicegeomdata::gNavIndex,
                         sizeof(vecgeom::globaldevicegeomdata::gNavIndex),
                         0,
                         CELER_DEVICE_API_SYMBOL(MemcpyDeviceToHost)));
    CELER_DEVICE_API_CALL(DeviceSynchronize());

    return result;
}

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
