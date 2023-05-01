//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/alongstep/AlongStepRZMapFieldMscAction.cu
//---------------------------------------------------------------------------//
#include "AlongStepRZMapFieldMscAction.hh"

#include "corecel/device_runtime_api.h"
#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/KernelParamCalculator.device.hh"
#include "celeritas/em/FluctuationParams.hh"
#include "celeritas/em/UrbanMscParams.hh"
#include "celeritas/field/RZMapFieldParams.hh"

#include "AlongStepLauncher.hh"
#include "detail/AlongStepRZMapFieldMsc.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
__global__ void
along_step_field_msc_kernel(DeviceCRef<CoreParamsData> const params,
                            DeviceRef<CoreStateData> const state,
                            DeviceCRef<UrbanMscData> const msc_data,
                            DeviceCRef<RZMapFieldParamsData> const field_data,
                            DeviceCRef<FluctuationData> const fluct)
{
    auto tid = KernelParamCalculator::thread_id();
    if (!(tid < state.size()))
        return;

    auto launch = make_along_step_launcher(params,
                                           state,
                                           msc_data,
                                           field_data,
                                           fluct,
                                           detail::along_step_mapfield_msc);
    launch(tid);
}
//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Launch the along-step action on device.
 */
void AlongStepRZMapFieldMscAction::execute(ParamsDeviceCRef const& params,
                                           StateDeviceRef& state) const
{
    CELER_EXPECT(params && state);
    CELER_LAUNCH_KERNEL(along_step_field_msc,
                        celeritas::device().default_block_size(),
                        state.size(),
                        params,
                        state,
                        msc_->device_ref(),
                        field_->device_ref(),
                        fluct_->device_ref());
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
