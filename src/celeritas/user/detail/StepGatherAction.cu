//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/detail/StepGatherAction.cu
//---------------------------------------------------------------------------//
#include "corecel/Macros.hh"
#include "corecel/sys/KernelParamCalculator.device.hh"

#include "../StepData.hh"
#include "StepGatherLauncher.hh"

namespace celeritas
{
namespace detail
{
namespace
{
//---------------------------------------------------------------------------//
template<StepPoint P>
__global__ void step_gather_kernel(CoreDeviceRef const              core,
                                   DeviceCRef<StepParamsData> const step_params,
                                   DeviceRef<StepStateData> const   step_state)
{
    auto tid = KernelParamCalculator::thread_id();
    if (!(tid < step_state.size()))
        return;

    StepGatherLauncher<P> launch{core, step_params, step_state};
    launch(tid);
}
//---------------------------------------------------------------------------//
} // namespace

//---------------------------------------------------------------------------//
/*!
 * Launch the action on device.
 */
template<StepPoint P>
void step_gather_device(CoreRef<MemSpace::device> const&  core,
                        DeviceCRef<StepParamsData> const& step_params,
                        DeviceRef<StepStateData> const&   step_state)
{
    CELER_EXPECT(core);
    CELER_EXPECT(step_state.size() == core.states.size());

    static const KernelParamCalculator calc_launch_params_(
        P == StepPoint::pre ? "step_gather_pre" : "step_gather_post",
        step_gather_kernel<P>);
    auto grid = calc_launch_params_(core.states.size());

    CELER_LAUNCH_KERNEL_IMPL(step_gather_kernel<P>,
                             grid.blocks_per_grid,
                             grid.threads_per_block,
                             0,
                             0,
                             core,
                             step_params,
                             step_state);
    CELER_DEVICE_CHECK_ERROR();
}

//---------------------------------------------------------------------------//

template void
step_gather_device<StepPoint::pre>(CoreRef<MemSpace::device> const&,
                                   DeviceCRef<StepParamsData> const&,
                                   DeviceRef<StepStateData> const&);
template void
step_gather_device<StepPoint::post>(CoreRef<MemSpace::device> const&,
                                    DeviceCRef<StepParamsData> const&,
                                    DeviceRef<StepStateData> const&);

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
