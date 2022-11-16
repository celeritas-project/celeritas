//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/detail/StepGatherAction.cu
//---------------------------------------------------------------------------//
#include "StepGatherAction.hh"

#include "corecel/Macros.hh"
#include "corecel/sys/KernelParamCalculator.device.hh"

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
void StepGatherAction<P>::execute(CoreDeviceRef const& core) const
{
    CELER_EXPECT(core);

    const auto& step_state = this->get_state(core);
    CELER_ASSERT(step_state.size() == core.states.size());

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
                             storage_->params.device_ref(),
                             step_state);
    CELER_DEVICE_CHECK_ERROR();

    if (P == StepPoint::post)
    {
        (*callback_)(step_state);
    }
}

//---------------------------------------------------------------------------//

template class StepGatherAction<StepPoint::pre>;
template class StepGatherAction<StepPoint::post>;

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
