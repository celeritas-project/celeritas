//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/geo/HeuristicGeoTestBase.cu
//---------------------------------------------------------------------------//
#include "corecel/DeviceRuntimeApi.hh"

#include "corecel/Types.hh"
#include "corecel/data/Filler.device.t.hh"
#include "corecel/sys/KernelLauncher.device.hh"
#include "corecel/sys/KernelParamCalculator.device.hh"

#include "HeuristicGeoExecutor.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Run on device and return results
void heuristic_test_execute(DeviceCRef<HeuristicGeoParamsData> const& params,
                            DeviceRef<HeuristicGeoStateData> const& state)
{
    HeuristicGeoExecutor execute_thread{params, state};
    static KernelLauncher<decltype(execute_thread)> const launch_kernel(
        "heuristic-geo");
    launch_kernel(state.size(), StreamId{}, execute_thread);

    CELER_DEVICE_CALL_PREFIX(DeviceSynchronize());
}

//---------------------------------------------------------------------------//
}  // namespace test

// Used by "resize" for state
template class Filler<::celeritas::test::LifeStatus, MemSpace::device>;

}  // namespace celeritas
