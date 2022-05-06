//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/generated/AlongStepAction.cu
//! \note Auto-generated by gen-action.py: DO NOT MODIFY!
//---------------------------------------------------------------------------//
#include "AlongStepAction.hh"

#include "corecel/device_runtime_api.h"
#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/sys/KernelParamCalculator.device.hh"
#include "corecel/sys/Device.hh"
#include "celeritas/global/TrackLauncher.hh"
#include "celeritas/global/detail/AlongStepActionImpl.hh"

namespace celeritas
{
namespace generated
{
namespace
{
__global__ void along_step_kernel(CoreDeviceRef const data
)
{
    auto tid = KernelParamCalculator::thread_id();
    if (!(tid < data.states.size()))
        return;

    auto launch = make_track_launcher(data, detail::along_step_track);
    launch(tid);
}
} // namespace

void AlongStepAction::execute(const CoreDeviceRef& data) const
{
    CELER_EXPECT(data);
    CELER_LAUNCH_KERNEL(along_step,
                        celeritas::device().default_block_size(),
                        data.states.size(),
                        data);
}

} // namespace generated
} // namespace celeritas
