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
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/em/launcher/SeltzerBergerLauncher.hh"
#include "celeritas/phys/InteractionLauncher.hh"

namespace celeritas
{
namespace generated
{
namespace
{
__global__ void seltzer_berger_interact_kernel(
    celeritas::DeviceCRef<celeritas::CoreParamsData> const params,
    celeritas::DeviceRef<celeritas::CoreStateData> const state,
    celeritas::SeltzerBergerDeviceRef const model_data)
{
    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (!(tid < state.size()))
        return;

    auto launch = celeritas::make_interaction_launcher(
        params, state, celeritas::seltzer_berger_interact_track, model_data);
    launch(tid);
}
}  // namespace

void seltzer_berger_interact(
    celeritas::CoreParams const& params,
    celeritas::CoreState<MemSpace::device>& state,
    celeritas::SeltzerBergerDeviceRef const& model_data)
{
    CELER_EXPECT(model_data);

    CELER_LAUNCH_KERNEL(seltzer_berger_interact,
                        celeritas::device().default_block_size(),
                        state.size(),
                        params.ref<MemSpace::native>(), state.ref(), model_data);
}

}  // namespace generated
}  // namespace celeritas
