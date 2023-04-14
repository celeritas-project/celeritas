//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/generated/SeltzerBergerInteract.cu
//! \note Auto-generated by gen-interactor.py: DO NOT MODIFY!
//---------------------------------------------------------------------------//
#include "SeltzerBergerInteract.hh"

#include "corecel/device_runtime_api.h"
#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/sys/KernelParamCalculator.device.hh"
#include "corecel/sys/Device.hh"
#include "celeritas/em/launcher/SeltzerBergerLauncher.hh"
#include "celeritas/phys/InteractionLauncher.hh"

namespace celeritas
{
namespace generated
{
namespace
{
__global__ void seltzer_berger_interact_kernel(
    celeritas::SeltzerBergerDeviceRef const model_data,
    celeritas::DeviceCRef<celeritas::CoreParamsData> const params,
    celeritas::DeviceRef<celeritas::CoreStateData> const state)
{
    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (!(tid < state.size()))
        return;

    auto launch = celeritas::make_interaction_launcher(
        params, state, model_data,
        celeritas::seltzer_berger_interact_track);
    launch(tid);
}
}  // namespace

void seltzer_berger_interact(
    celeritas::SeltzerBergerDeviceRef const& model_data,
    celeritas::DeviceCRef<celeritas::CoreParamsData> const& params,
    celeritas::DeviceRef<celeritas::CoreStateData>& state)
{
    CELER_EXPECT(params && state);
    CELER_EXPECT(model_data);

    CELER_LAUNCH_KERNEL(seltzer_berger_interact,
                        celeritas::device().default_block_size(),
                        state.size(),
                        model_data, params, state);
}

}  // namespace generated
}  // namespace celeritas
