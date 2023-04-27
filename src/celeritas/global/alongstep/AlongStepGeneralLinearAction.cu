//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
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
#include "celeritas/global/TrackLauncher.hh"

#include "detail/AlongStepGeneralLinear.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
__global__ void
along_step_general_linear_kernel(DeviceCRef<CoreParamsData> const params,
                                 DeviceRef<CoreStateData> const state,
                                 DeviceCRef<UrbanMscData> const msc_params,
                                 DeviceCRef<FluctuationData> const fluct)
{
    auto launch = make_alive_track_launcher(
        params, state, detail::along_step_general_linear, msc_params, fluct);
    launch(tid);
}
//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Launch the along-step action on device.
 */
void AlongStepGeneralLinearAction::execute(ParamsDeviceCRef const& params,
                                           StateDeviceRef& state) const
{
    CELER_EXPECT(params && state);
    CELER_LAUNCH_KERNEL(along_step_general_linear,
                        celeritas::device().default_block_size(),
                        state.size(),
                        params,
                        state,
                        device_data_.msc,
                        device_data_.fluct);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
