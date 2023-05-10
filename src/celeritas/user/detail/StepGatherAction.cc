//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
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
#include "corecel/sys/MultiExceptionHandler.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/CoreTrackData.hh"
#include "celeritas/user/StepData.hh"

#include "StepGatherLauncher.hh"

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
    auto const& step_state
        = storage_->obj.state<MemSpace::host>(state.stream_id(), state.size());

    MultiExceptionHandler capture_exception;
    StepGatherLauncher<P> launch{params.ref<MemSpace::native>(),
                                 state.ref(),
                                 storage_->obj.params<MemSpace::host>(),
                                 step_state};
#pragma omp parallel for
    for (size_type i = 0; i < state.size(); ++i)
    {
        CELER_TRY_HANDLE(launch(ThreadId{i}), capture_exception);
    }
    log_and_rethrow(std::move(capture_exception));

    if (P == StepPoint::post)
    {
        StepState<MemSpace::host> cb_state{step_state, state.stream_id()};
        for (auto const& sp_callback : callbacks_)
        {
            sp_callback->process_steps(cb_state);
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Gather step attributes from GPU data, and execute callbacks at end of step.
 */
template<StepPoint P>
void StepGatherAction<P>::execute(CoreParams const& params,
                                  CoreStateDevice& state) const
{
    auto& step_state = storage_->obj.state<MemSpace::device>(state.stream_id(),
                                                             state.size());
    step_gather_device<P>(params.ref<MemSpace::device>(),
                          state.ref(),
                          storage_->obj.params<MemSpace::device>(),
                          step_state);

    if (P == StepPoint::post)
    {
        StepState<MemSpace::device> cb_state{step_state, state.stream_id()};
        for (auto const& sp_callback : callbacks_)
        {
            sp_callback->process_steps(cb_state);
        }
    }
}

//---------------------------------------------------------------------------//
// FREE FUNCTIONS INSTANTIATION
//---------------------------------------------------------------------------//

#if !CELER_USE_DEVICE
template<StepPoint P>
void step_gather_device(DeviceCRef<CoreParamsData> const&,
                        DeviceRef<CoreStateData>&,
                        DeviceCRef<StepParamsData> const&,
                        DeviceRef<StepStateData>&)
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
