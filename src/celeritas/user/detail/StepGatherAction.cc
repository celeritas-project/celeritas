//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/detail/StepGatherAction.cc
//---------------------------------------------------------------------------//
#include "StepGatherAction.hh"

#include <mutex>
#include <utility>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/cont/Range.hh"
#include "corecel/io/Logger.hh"
#include "celeritas/global/ActionLauncher.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/CoreTrackData.hh"
#include "celeritas/global/TrackExecutor.hh"
#include "celeritas/user/StepData.hh"

#include "StepGatherExecutor.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Capture construction arguments.
 */
template<StepPoint P>
StepGatherAction<P>::StepGatherAction(ActionId id,
                                      SPStepStorage storage,
                                      VecInterface callbacks)
    : id_(id), storage_(std::move(storage)), callbacks_(std::move(callbacks))
{
    CELER_EXPECT(id_);
    CELER_EXPECT(!callbacks_.empty() || P == StepPoint::pre);
    CELER_EXPECT(storage_);
}

//---------------------------------------------------------------------------//
/*!
 * Descriptive name of the action.
 */
template<StepPoint P>
std::string StepGatherAction<P>::description() const
{
    std::string result = "gather ";
    result += P == StepPoint::pre ? "pre" : "post";
    result += "-step steps/hits";
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Gather step attributes from host data, and execute callbacks at end of step.
 */
template<StepPoint P>
void StepGatherAction<P>::execute(CoreParams const& params,
                                  CoreStateHost& state) const
{
    auto const& step_state = storage_->obj.state<MemSpace::native>(
        state.stream_id(), state.size());
    auto execute = TrackExecutor{
        params.ptr<MemSpace::native>(),
        state.ptr(),
        detail::StepGatherExecutor<P>{storage_->obj.params<MemSpace::native>(),
                                      step_state}};
    launch_action(*this, params, state, execute);

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
#if !CELER_USE_DEVICE
template<StepPoint P>
void StepGatherAction<P>::execute(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//

template class StepGatherAction<StepPoint::pre>;
template class StepGatherAction<StepPoint::post>;

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
