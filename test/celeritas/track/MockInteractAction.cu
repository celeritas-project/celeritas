//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/MockInteractAction.cu
//---------------------------------------------------------------------------//
#include "MockInteractAction.hh"

#include "corecel/device_runtime_api.h"
#include "corecel/sys/Device.hh"
#include "corecel/sys/KernelParamCalculator.device.hh"
#include "celeritas/global/TrackLauncher.hh"

#include "MockInteractImpl.hh"

namespace celeritas
{
namespace test
{
namespace
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//

__global__ void mock_interact_kernel(DeviceCRef<CoreParamsData> const params,
                                     DeviceRef<CoreStateData> const state,
                                     DeviceCRef<MockInteractData> const input)
{
    auto launch = make_active_track_launcher(
        params, state, apply_mock_interact, input);
    launch(KernelParamCalculator::thread_id());
}
//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//

void MockInteractAction::execute(ParamsDeviceCRef const& params,
                                 StateDeviceRef& state) const
{
    CELER_EXPECT(params && state);
    CELER_EXPECT(state.size() == data_.device_ref().size());

    CELER_LAUNCH_KERNEL(mock_interact,
                        device().default_block_size(),
                        state.size(),
                        params,
                        state,
                        data_.device_ref());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
