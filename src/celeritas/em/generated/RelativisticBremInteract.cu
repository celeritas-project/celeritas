//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/generated/RelativisticBremInteract.cu
//! \note Auto-generated by gen-interactor.py: DO NOT MODIFY!
//---------------------------------------------------------------------------//
#include "RelativisticBremInteract.hh"

#include "corecel/device_runtime_api.h"
#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/sys/KernelParamCalculator.device.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/Stream.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/em/launcher/RelativisticBremLauncher.hh"
#include "celeritas/phys/InteractionLauncher.hh"

using celeritas::MemSpace;

namespace celeritas
{
namespace generated
{
namespace
{
__global__ void relativistic_brem_interact_kernel(
    celeritas::CRefPtr<celeritas::CoreParamsData, MemSpace::device> const params,
    celeritas::RefPtr<celeritas::CoreStateData, MemSpace::device> const state,
    celeritas::RelativisticBremDeviceRef const model_data,
    celeritas::size_type size)
{
    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (!(tid < size))
        return;

    auto launch = celeritas::make_interaction_launcher(
        params, state, celeritas::relativistic_brem_interact_track, model_data);
    launch(tid);
}
}  // namespace

void relativistic_brem_interact(
    celeritas::CoreParams const& params,
    celeritas::CoreState<MemSpace::device>& state,
    celeritas::RelativisticBremDeviceRef const& model_data)
{
    CELER_EXPECT(model_data);

    CELER_LAUNCH_KERNEL(relativistic_brem_interact,
                        celeritas::device().default_block_size(),
                        state.size(),
                        celeritas::device().stream(state.stream_id()).get(),
                        params.ptr<MemSpace::native>(),
                        state.ptr(),
                        model_data,
                        state.size());
}

}  // namespace generated
}  // namespace celeritas
