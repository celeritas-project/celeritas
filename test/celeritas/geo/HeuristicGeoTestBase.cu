//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/geo/HeuristicGeoTestBase.cu
//---------------------------------------------------------------------------//
#include "corecel/device_runtime_api.h"
#include "corecel/Types.hh"
#include "corecel/data/detail/Filler.device.t.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/KernelParamCalculator.device.hh"

#include "HeuristicGeoLauncher.hh"

namespace celeritas
{
namespace test
{
namespace
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//
__global__ void
heuristic_test_kernel(DeviceCRef<HeuristicGeoParamsData> const params,
                      DeviceRef<HeuristicGeoStateData> const state)
{
    auto tid = KernelParamCalculator::thread_id();
    if (tid.get() >= state.size())
        return;

    HeuristicGeoLauncher launch{params, state};
    launch(tid);
}
}  // namespace

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Run on device and return results
void heuristic_test_launch(DeviceCRef<HeuristicGeoParamsData> const& params,
                           DeviceRef<HeuristicGeoStateData> const& state)
{
    CELER_LAUNCH_KERNEL(heuristic_test,
                        device().default_block_size(),
                        state.size(),
                        params,
                        state);

    CELER_DEVICE_CALL_PREFIX(DeviceSynchronize());
}

//---------------------------------------------------------------------------//
}  // namespace test

namespace detail
{
template class Filler<::celeritas::test::LifeStatus, MemSpace::device>;
}
}  // namespace celeritas
