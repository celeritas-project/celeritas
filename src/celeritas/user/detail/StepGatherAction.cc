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
void step_gather_device(DeviceCRef<CoreParamsData> const& core_params,
                        DeviceRef<CoreStateData>& core_state,
                        DeviceCRef<StepParamsData> const& step_params,
                        DeviceRef<StepStateData>& step_state);

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
void StepGatherAction<P>::execute(ParamsHostCRef const& params,
                                  StateHostRef& states) const
{
    CELER_EXPECT(params && states);
    auto const& step_state = this->get_state(params, states);
    CELER_ASSERT(step_state.size() == states.size());

    MultiExceptionHandler capture_exception;
    StepGatherLauncher<P> launch{
        params, states, storage_->params.host_ref(), step_state};
#pragma omp parallel for
    for (size_type i = 0; i < states.size(); ++i)
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
void StepGatherAction<P>::execute(ParamsDeviceCRef const& params,
                                  StateDeviceRef& states) const
{
    CELER_EXPECT(params && states);

#if CELER_USE_DEVICE
    auto& step_state = this->get_state(params, states);
    step_gather_device<P>(
        params, states, storage_->params.device_ref(), step_state);

    if (P == StepPoint::post)
    {
        for (auto const& sp_callback : callbacks_)
        {
            sp_callback->execute(step_state);
        }
    }
#else
    CELER_NOT_CONFIGURED("CUDA OR HIP");
#endif
}

//---------------------------------------------------------------------------//
/*!
 * Get a reference to the stream-local step state data, allocating if needed.
 *
 * This is thread-safe and allocates storage on demand for each stream *and*
 * device type.
 */
template<MemSpace M>
StepStateData<Ownership::reference, M>&
get_stream_state(CoreParamsData<Ownership::const_reference, M> const& params,
                 CoreStateData<Ownership::reference, M>& states,
                 StepStorage* storage)
{
    CELER_EXPECT(storage);

    auto& state_vec = storage->get_states<M>();
    if (CELER_UNLIKELY(state_vec.empty()))
    {
        // State storage hasn't been resized to the number of streams yet:
        // mutex and resize if needed
        static std::mutex resize_mutex;
        std::lock_guard<std::mutex> scoped_lock{resize_mutex};
        if (state_vec.empty())
        {
            // State is guaranteed unresized and we've got a lock
            CELER_LOG_LOCAL(debug)
                << "Resizing " << (M == MemSpace::host ? "host" : "device")
                << " step state data for " << params.scalars.max_streams
                << " threads";
            state_vec.resize(params.scalars.max_streams);
        }
    }

    // Get the stream-local but possibly unallocated state storage for the
    // current stream
    CELER_ASSERT(states.stream_id < state_vec.size());
    auto& state_store = state_vec[states.stream_id.unchecked_get()];
    if (CELER_UNLIKELY(!state_store))
    {
        // Thread-local data hasn't been allocated yet
        CELER_LOG_LOCAL(debug)
            << "Allocating local " << (M == MemSpace::host ? "host" : "device")
            << " step state data";
        state_store = CollectionStateStore<StepStateData, M>{
            storage->params.host_ref(), states.size()};
    }

    CELER_ENSURE(state_store);
    return state_store.ref();
}

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//

template class StepGatherAction<StepPoint::pre>;
template class StepGatherAction<StepPoint::post>;

template StepStateData<Ownership::reference, MemSpace::host>&
get_stream_state(HostCRef<CoreParamsData> const&,
                 HostRef<CoreStateData>&,
                 StepStorage*);
template StepStateData<Ownership::reference, MemSpace::device>&
get_stream_state(DeviceCRef<CoreParamsData> const&,
                 DeviceRef<CoreStateData>&,
                 StepStorage*);

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
