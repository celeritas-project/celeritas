//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/alongstep/AlongStepUniformMscAction.cu
//---------------------------------------------------------------------------//
#include "AlongStepUniformMscAction.hh"

#include "corecel/device_runtime_api.h"
#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/KernelParamCalculator.device.hh"

#include "AlongStepLauncher.hh"
#include "detail/AlongStepUniformMsc.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
__global__ void
along_step_uniform_msc_kernel(CoreRef<MemSpace::device> const track_data,
                              DeviceCRef<UrbanMscData> const msc_data,
                              UniformFieldParams const field_params)
{
    auto tid = KernelParamCalculator::thread_id();
    if (!(tid < track_data.states.size()))
        return;

    auto launch = make_along_step_launcher(track_data,
                                           msc_data,
                                           field_params,
                                           NoData{},
                                           detail::along_step_uniform_msc);
    launch(tid);
}
//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Launch the along-step action on device.
 */
void AlongStepUniformMscAction::execute(ParamsDeviceCRef const& params,
                                        StateDeviceRef& states) const
{
    CELER_EXPECT(data);
    CELER_LAUNCH_KERNEL(along_step_uniform_msc,
                        celeritas::device().default_block_size(),
                        data.states.size(),
                        data,
                        device_data_.msc,
                        field_params_);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
