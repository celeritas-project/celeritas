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
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"

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

__global__ void
mock_interact_kernel(CRefPtr<CoreParamsData, MemSpace::device> const params,
                     RefPtr<CoreStateData, MemSpace::device> const state,
                     DeviceCRef<MockInteractData> const input)
{
    auto execute = make_active_track_executor(
        params, state, apply_mock_interact, input);
    execute(KernelParamCalculator::thread_id());
}
//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//

void MockInteractAction::execute(CoreParams const& params,
                                 CoreStateDevice& state) const
{
    CELER_EXPECT(state.size() == data_.device_ref().size());

    CELER_LAUNCH_KERNEL(mock_interact,
                        device().default_block_size(),
                        state.size(),
                        0,
                        params.ptr<MemSpace::native>(),
                        state.ptr(),
                        data_.device_ref());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
