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
#include "celeritas/global/CoreTrackData.hh"
#include "celeritas/user/StepData.hh"

#include "StepGatherLauncher.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
template<StepPoint P>
void step_gather_device(CoreRef<MemSpace::device> const&,
                        DeviceCRef<StepParamsData> const&,
                        DeviceRef<StepStateData> const&)
#if CELER_USE_DEVICE
    ;
#else
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

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
    return P == StepPoint::pre    ? "pre-step state gather"
           : P == StepPoint::post ? "post-step state gather"
                                  : "";
}

//---------------------------------------------------------------------------//
/*!
 * Gather step attributes from host data, and execute callbacks at end of step.
 */
template<StepPoint P>
void StepGatherAction<P>::execute(CoreHostRef const& core) const
{
    CELER_EXPECT(core);
    auto& state = core.states;
    auto const& step_state
        = storage_->obj.state<MemSpace::host>(state.stream_id, state.size());

    MultiExceptionHandler capture_exception;
    StepGatherLauncher<P> launch{
        core, storage_->obj.params<MemSpace::host>(), step_state};
#pragma omp parallel for
    for (size_type i = 0; i < core.states.size(); ++i)
    {
        CELER_TRY_HANDLE(launch(ThreadId{i}), capture_exception);
    }
    log_and_rethrow(std::move(capture_exception));

    if (P == StepPoint::post)
    {
        for (auto const& sp_callback : callbacks_)
        {
            sp_callback->execute(step_state);
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Gather step attributes from GPU data, and execute callbacks at end of step.
 */
template<StepPoint P>
void StepGatherAction<P>::execute(CoreDeviceRef const& core) const
{
    CELER_EXPECT(core);

    auto& state = core.states;
    auto& step_state
        = storage_->obj.state<MemSpace::device>(state.stream_id, state.size());
    step_gather_device<P>(
        core, storage_->obj.params<MemSpace::device>(), step_state);

    if (P == StepPoint::post)
    {
        for (auto const& sp_callback : callbacks_)
        {
            sp_callback->execute(step_state);
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
