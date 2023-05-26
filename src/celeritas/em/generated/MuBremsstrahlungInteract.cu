//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/generated/MuBremsstrahlungInteract.cu
//! \note Auto-generated by gen-interactor.py: DO NOT MODIFY!
//---------------------------------------------------------------------------//
#include "MuBremsstrahlungInteract.hh"

#include "corecel/device_runtime_api.h"
#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/sys/KernelParamCalculator.device.hh"
#include "corecel/sys/Device.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/KernelLaunchUtils.hh"
#include "celeritas/em/launcher/MuBremsstrahlungLauncher.hh"
#include "celeritas/phys/InteractionLauncher.hh"

using celeritas::MemSpace;

namespace celeritas
{
namespace generated
{
namespace
{
__global__ void mu_bremsstrahlung_interact_kernel(
    celeritas::CRefPtr<celeritas::CoreParamsData, MemSpace::device> const params,
    celeritas::RefPtr<celeritas::CoreStateData, MemSpace::device> const state,
    celeritas::MuBremsstrahlungDeviceRef const model_data,
    celeritas::size_type size,
    celeritas::ThreadId const offset)
{
    auto tid = celeritas::KernelParamCalculator::thread_id() + offset.get();
    if (!(tid < size))
        return;

    auto launch = celeritas::make_interaction_launcher(
        params, state, celeritas::mu_bremsstrahlung_interact_track, model_data);
    launch(tid);
}
}  // namespace

void mu_bremsstrahlung_interact(
    celeritas::CoreParams const& params,
    celeritas::CoreState<MemSpace::device>& state,
    celeritas::MuBremsstrahlungDeviceRef const& model_data,
    celeritas::ActionId action)
{
    CELER_EXPECT(model_data);
    KernelLaunchParams kernel_params = compute_launch_params(action,
                                                             params,
                                                             state,
                                                             TrackOrder::sort_step_limit_action);
    if (!kernel_params.num_threads)
        return;
    CELER_LAUNCH_KERNEL(mu_bremsstrahlung_interact,
                        celeritas::device().default_block_size(),
                        kernel_params.num_threads,
                        params.ptr<MemSpace::native>(),
                        state.ptr(),
                        model_data,
                        kernel_params.num_threads,
                        kernel_params.threads_offset);
}

}  // namespace generated
}  // namespace celeritas
