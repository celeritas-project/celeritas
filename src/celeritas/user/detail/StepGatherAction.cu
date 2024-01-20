//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/detail/StepGatherAction.cu
//---------------------------------------------------------------------------//
#include "StepGatherAction.hh"

#include "corecel/Macros.hh"
#include "celeritas/global/ActionLauncher.device.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"

#include "../StepData.hh"
#include "StepGatherExecutor.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Gather step attributes from GPU data, and execute callbacks at end of step.
 */
template<StepPoint P>
void StepGatherAction<P>::execute(CoreParams const& params,
                                  CoreStateDevice& state) const
{
    auto& step_state = storage_->obj.state<MemSpace::native>(state.stream_id(),
                                                             state.size());
    auto execute = TrackExecutor{
        params.ptr<MemSpace::native>(),
        state.ptr(),
        detail::StepGatherExecutor<P>{storage_->obj.params<MemSpace::native>(),
                                      step_state}};
    static ActionLauncher<decltype(execute)> const launch_kernel(*this);
    launch_kernel(state, execute);

    if (P == StepPoint::post)
    {
        StepState<MemSpace::native> cb_state{step_state, state.stream_id()};
        for (auto const& sp_callback : callbacks_)
        {
            sp_callback->process_steps(cb_state);
        }
    }
}

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//

template class StepGatherAction<StepPoint::pre>;
template class StepGatherAction<StepPoint::post>;

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
