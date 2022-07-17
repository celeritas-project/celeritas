//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/alongstep/AlongStepNeutralAction.cu
//---------------------------------------------------------------------------//
#include "AlongStepNeutralAction.hh"

#include "corecel/device_runtime_api.h"
#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/KernelParamCalculator.device.hh"

#include "AlongStepLauncher.hh"
#include "detail/AlongStepNeutral.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
__global__ void along_step_neutral_kernel(CoreDeviceRef const data)
{
    auto tid = KernelParamCalculator::thread_id();
    if (!(tid < data.states.size()))
        return;

    auto launch = make_along_step_launcher(
        data, NoData{}, NoData{}, NoData{}, detail::along_step_neutral);
    launch(tid);
}
//---------------------------------------------------------------------------//
} // namespace

//---------------------------------------------------------------------------//
/*!
 * Launch the along-step action on device.
 */
void AlongStepNeutralAction::execute(const CoreDeviceRef& data) const
{
    CELER_EXPECT(data);
    CELER_LAUNCH_KERNEL(along_step_neutral,
                        celeritas::device().default_block_size(),
                        data.states.size(),
                        data);
}

//---------------------------------------------------------------------------//
} // namespace celeritas
