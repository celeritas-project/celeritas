//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/alongstep/AlongStepGeneralLinearAction.cu
//---------------------------------------------------------------------------//
#include "AlongStepGeneralLinearAction.hh"

#include "corecel/device_runtime_api.h"
#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/KernelParamCalculator.device.hh"

#include "AlongStepLauncher.hh"
#include "detail/AlongStepGeneralLinear.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
__global__ void
along_step_general_linear_kernel(CoreRef<MemSpace::device> const track_data,
                                 DeviceCRef<UrbanMscData> const msc_data,
                                 DeviceCRef<FluctuationData> const fluct)
{
    auto tid = KernelParamCalculator::thread_id();
    if (!(tid < track_data.states.size()))
        return;

    auto launch = make_along_step_launcher(track_data,
                                           msc_data,
                                           NoData{},
                                           fluct,
                                           detail::along_step_general_linear);
    launch(tid);
}
//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Launch the along-step action on device.
 */
void AlongStepGeneralLinearAction::execute(CoreDeviceRef const& data) const
{
    CELER_EXPECT(data);
    CELER_LAUNCH_KERNEL(along_step_general_linear,
                        celeritas::device().default_block_size(),
                        data.states.size(),
                        data,
                        device_data_.msc,
                        device_data_.fluct);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
