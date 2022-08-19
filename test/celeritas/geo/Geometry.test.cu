//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/geo/Geometry.test.cu
//---------------------------------------------------------------------------//
#include "Geometry.test.hh"

#include "corecel/device_runtime_api.h"
#include "corecel/Types.hh"
#include "corecel/data/detail/Filler.device.t.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/KernelParamCalculator.device.hh"

namespace celeritas
{
namespace test
{
namespace
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//

__global__ void g_test_kernel(const DeviceCRef<GeoTestParamsData> params,
                              const DeviceRef<GeoTestStateData>   state)
{
    auto tid = KernelParamCalculator::thread_id();
    if (tid.get() >= state.size())
        return;

    GeoTestLauncher launch{params, state};
    launch(tid);
}
} // namespace

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Run on device and return results
void g_test(const DeviceCRef<GeoTestParamsData>& params,
            const DeviceRef<GeoTestStateData>&   state)
{
    CELER_LAUNCH_KERNEL(
        g_test, device().default_block_size(), state.size(), params, state);

    CELER_DEVICE_CALL_PREFIX(DeviceSynchronize());
}

//---------------------------------------------------------------------------//
} // namespace test

namespace detail
{
template class Filler<::celeritas::test::LifeStatus, MemSpace::device>;
}
} // namespace celeritas
