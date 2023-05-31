//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/detail/StepGatherAction.cu
//---------------------------------------------------------------------------//
#include "corecel/Macros.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/KernelParamCalculator.device.hh"
#include "corecel/sys/Stream.hh"

#include "../StepData.hh"
#include "StepGatherExecutor.hh"

namespace celeritas
{
namespace detail
{
namespace
{
//---------------------------------------------------------------------------//
template<StepPoint P>
__global__ void
step_gather_kernel(DeviceCRef<CoreParamsData> const core_params,
                   DeviceRef<CoreStateData> const core_state,
                   DeviceCRef<StepParamsData> const step_params,
                   DeviceRef<StepStateData> const step_state)
{
    auto tid = KernelParamCalculator::thread_id();
    if (!(tid < step_state.size()))
        return;

    StepGatherExecutor<P> execute{
        core_params, core_state, step_params, step_state};
    execute(tid);
}
//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Launch the action on device.
 */
template<StepPoint P>
void step_gather_device(DeviceCRef<CoreParamsData> const& core_params,
                        DeviceRef<CoreStateData>& core_state,
                        DeviceCRef<StepParamsData> const& step_params,
                        DeviceRef<StepStateData>& step_state)
{
    CELER_EXPECT(core_params && core_state);
    CELER_EXPECT(step_state.size() == core_state.size());

    static const KernelParamCalculator calc_launch_params_(
        P == StepPoint::pre ? "step_gather_pre" : "step_gather_post",
        step_gather_kernel<P>);
    auto grid = calc_launch_params_(core_state.size());

    CELER_LAUNCH_KERNEL_IMPL(
        step_gather_kernel<P>,
        grid.blocks_per_grid,
        grid.threads_per_block,
        0,
        celeritas::device().stream(core_state.stream_id).get(),
        core_params,
        core_state,
        step_params,
        step_state);
    CELER_DEVICE_CHECK_ERROR();
}

//---------------------------------------------------------------------------//

template void
step_gather_device<StepPoint::pre>(DeviceCRef<CoreParamsData> const&,
                                   DeviceRef<CoreStateData>&,
                                   DeviceCRef<StepParamsData> const&,
                                   DeviceRef<StepStateData>&);
template void
step_gather_device<StepPoint::post>(DeviceCRef<CoreParamsData> const&,
                                    DeviceRef<CoreStateData>&,
                                    DeviceCRef<StepParamsData> const&,
                                    DeviceRef<StepStateData>&);

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
