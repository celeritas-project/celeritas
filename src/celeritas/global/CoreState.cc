//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/CoreState.cc
//---------------------------------------------------------------------------//
#include "CoreState.hh"

#include "corecel/Macros.hh"
#ifdef CELER_USE_DEVICE
#    include "corecel/data/ObserverPtr.device.hh"
#endif
#include "corecel/io/Logger.hh"
#include "corecel/sys/ActionRegistry.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "celeritas/track/TrackInitParams.hh"

#include "CoreParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Support polymorphic deletion
CoreStateInterface::~CoreStateInterface() = default;

//---------------------------------------------------------------------------//
/*!
 * Construct from CoreParams.
 */
template<MemSpace M>
CoreState<M>::CoreState(CoreParams const& params, StreamId stream_id)
    : CoreState{params, stream_id, params.tracks_per_stream()}
{
}

//---------------------------------------------------------------------------//
/*!
 * Construct with manual slot count.
 *
 * This is currently used for unit tests, and temporarily used by the \c
 * Stepper constructor.
 */
template<MemSpace M>
CoreState<M>::CoreState(CoreParams const& params,
                        StreamId stream_id,
                        size_type num_track_slots)
{
    CELER_VALIDATE(stream_id < params.max_streams(),
                   << "stream ID " << stream_id.unchecked_get()
                   << " is out of range: max streams is "
                   << params.max_streams());
    CELER_VALIDATE(num_track_slots > 0, << "number of track slots is not set");

    ScopedProfiling profile_this{"construct-state"};

    states_ = CollectionStateStore<CoreStateData, M>(
        params.host_ref(), stream_id, num_track_slots);

    if constexpr (M == MemSpace::device)
    {
        auto counters = CoreStateCounters{};
        counters.num_vacancies = num_track_slots;
        this->sync_put_counters(counters);
        device_ref_vec_ = DeviceVector<Ref>(1);
        device_ref_vec_.copy_to_device({&this->ref(), 1});
        ptr_ = make_observer(device_ref_vec_);
    }
    else if constexpr (M == MemSpace::host)
    {
        auto& counters = this->counters();
        counters.num_vacancies = num_track_slots;
        ptr_ = make_observer(&this->ref());
    }

    if (params.aux_reg())
    {
        // Allocate auxiliary data
        aux_state_ = std::make_shared<AuxStateVec>(
            *params.aux_reg(), M, stream_id, num_track_slots);
    }

    if (is_action_sorted(params.init()->track_order()))
    {
        offsets_.resize(params.action_reg()->num_actions() + 1);
    }

    CELER_LOG(status) << "Celeritas core state initialization complete";
    CELER_ENSURE(states_);
    CELER_ENSURE(ptr_);
    CELER_ENSURE(aux_state_);
}

//---------------------------------------------------------------------------//
/*!
 * Print diagnostic when core state is being deleted.
 */
template<MemSpace M>
CoreState<M>::~CoreState()
{
    try
    {
        CELER_LOG(debug) << "Deallocating " << to_cstring(M)
                         << " core state (stream "
                         << this->stream_id().unchecked_get() << ')';
    }
    catch (...)  // NOLINT(bugprone-empty-catch)
    {
        // Ignore anything bad that happens while logging
    }
}

//---------------------------------------------------------------------------//
/*!
 * Whether the state should be transported with no active particles.
 *
 * This can only be called when there are no active tracks. It should be
 * immediately cleared after a step.
 *
 * \sa Stepper::warm_up
 */
template<MemSpace M>
void CoreState<M>::warming_up(bool new_state)
{
    size_type num_active;
    if constexpr (M == MemSpace::host)
    {
        auto& counters = this->counters();
        num_active = counters.num_active;
    }
    else if constexpr (M == MemSpace::device)
    {
        auto counters = this->sync_get_counters();
        num_active = counters.num_active;
    }
    CELER_EXPECT(!new_state || num_active == 0);
    warming_up_ = new_state;
}

//---------------------------------------------------------------------------//
/*!
 * Get a range of sorted track slots about to undergo a given action.
 *
 * The result delimits the [start, end) of the track partition assigned
 * \c action_id in track_slots.
 */
template<MemSpace M>
Range<ThreadId> CoreState<M>::get_action_range(ActionId action_id) const
{
    auto const& thread_offsets = offsets_.host_action_thread_offsets();
    CELER_EXPECT((action_id + 1) < thread_offsets.size());
    return {thread_offsets[action_id], thread_offsets[action_id + 1]};
}

#if CELER_USE_DEVICE
//---------------------------------------------------------------------------//
/*!
 * Copy the core state counters from the device to the host. Since the entire
 * sequence of actions in a step are performed on the device, this is typically
 * done at the end of a step.
 */
template<MemSpace M>
CoreStateCounters const CoreState<M>::sync_get_counters() const
{
    if constexpr (M == MemSpace::device)
    {
        auto counters = device_pointer_cast(this->ref().init.counters.data());
        return ItemCopier<CoreStateCounters>{stream_id()}(counters.get());
    }
    else if constexpr (M == MemSpace::host)
    {
        // return *(this->ref().init.counters.data().get());
        CELER_ASSERT_UNREACHABLE();
        return CoreStateCounters{};
    }
}

//---------------------------------------------------------------------------//
/*!
 * Copy the core state counters from the host to the device. This function is a
 * placeholder function until the corresponding host code that updates the Core
 * State counters can be moved to device functions.
 */
template<MemSpace M>
void CoreState<M>::sync_put_counters(CoreStateCounters& host_counters)
{
    if constexpr (M == MemSpace::device)
    {
        auto counters = device_pointer_cast(this->ref().init.counters.data());
        copy_bytes(MemSpace::device,
                   counters.get(),
                   MemSpace::host,
                   &host_counters,
                   sizeof(CoreStateCounters),
                   stream_id());
    }
    else if constexpr (M == MemSpace::host)
    {
        CELER_ASSERT_UNREACHABLE();
    }
    return;
}
#endif

//---------------------------------------------------------------------------//
/*!
 * Reset the state data.
 *
 * This clears the state counters and initializes the necessary state data so
 * the state can be reused for a new event. This should be necessary only if
 * the previous event aborted early.
 */
template<MemSpace M>
void CoreState<M>::reset()
{
    if constexpr (M == MemSpace::host)
    {
        auto& counters = this->counters();
        counters = CoreStateCounters{};
        counters.num_vacancies = this->size();
    }
    else if constexpr (M == MemSpace::device)
    {
        auto counters = CoreStateCounters{};
        counters.num_vacancies = this->size();
        sync_put_counters(counters);
    }

    // Reset all the track slots to inactive
    fill(TrackStatus::inactive, &this->ref().sim.status);

    // Mark all the track slots as empty
    fill_sequence(&this->ref().init.vacancies, this->stream_id());
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
template<MemSpace M>
CoreStateCounters const CoreState<M>::sync_get_counters() const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}

template<MemSpace M>
void CoreState<M>::sync_put_counters(CoreStateCounters& host_counters)
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//
template class CoreState<MemSpace::host>;
template class CoreState<MemSpace::device>;
//---------------------------------------------------------------------------//

}  // namespace celeritas
