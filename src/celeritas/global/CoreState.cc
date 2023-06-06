//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/CoreState.cc
//---------------------------------------------------------------------------//
#include "CoreState.hh"

#include "corecel/data/Copier.hh"
#include "celeritas/track/detail/TrackSortUtils.hh"
#include "corecel/io/Logger.hh"

#include "CoreParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from CoreParams.
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

    states_ = CollectionStateStore<CoreStateData, M>(
        params.host_ref(), stream_id, num_track_slots);

    counters_.num_vacancies = num_track_slots;
    counters_.num_primaries = 0;
    counters_.num_initializers = 0;

    if constexpr (M == MemSpace::device)
    {
        device_ref_vec_ = DeviceVector<Ref>(1);
        device_ref_vec_.copy_to_device({&this->ref(), 1});
    }

    CELER_LOG_LOCAL(status) << "Celeritas core state initialization complete";
    CELER_ENSURE(states_);
}

//---------------------------------------------------------------------------//
/*!
 * Inject primaries to be turned into TrackInitializers.
 *
 * These will be converted by the ProcessPrimaries action.
 */
template<MemSpace M>
void CoreState<M>::insert_primaries(Span<Primary const> host_primaries)
{
    // Copy primaries
    if (primaries_.size() < host_primaries.size())
    {
        primaries_ = {};
        resize(&primaries_, host_primaries.size());
    }
    counters_.num_primaries = host_primaries.size();

    Copier<Primary, M> copy_to_temp{primaries_[this->primary_range()]};
    copy_to_temp(MemSpace::host, host_primaries);
}

template<MemSpace M>
auto CoreState<M>::host_thread_offsets() -> ThreadItems<MemSpace::host>&
{
    if constexpr (M == MemSpace::device)
    {
        return host_thread_offsets_;
    }
    else
    {
        return thread_offsets_;
    }
}

template<MemSpace M>
void CoreState<M>::count_tracks_per_action(TrackOrder order)
{
    detail::count_tracks_per_action(states_.ref(),
                                    thread_offsets_[AllItems<ThreadId, M>{}],
                                    host_thread_offsets(),
                                    order);
}

template<>
Range<ThreadId>
CoreState<MemSpace::device>::get_action_range(ActionId action_id) const
{
    CELER_EXPECT((action_id + 1) < host_thread_offsets_.size());
    return {host_thread_offsets_[action_id],
            host_thread_offsets_[action_id + 1]};
}

template<>
Range<ThreadId>
CoreState<MemSpace::host>::get_action_range(ActionId action_id) const
{
    CELER_EXPECT((action_id + 1) < thread_offsets_.size());
    return {thread_offsets_[action_id], thread_offsets_[action_id + 1]};
}

template<MemSpace M>
void CoreState<M>::resize_offsets(size_type n)
{
    resize(&thread_offsets_, n);
    if constexpr (M == MemSpace::device)
    {
        resize(&host_thread_offsets_, n);
    }
}

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//
template class CoreState<MemSpace::host>;
template class CoreState<MemSpace::device>;
//---------------------------------------------------------------------------//
}  // namespace celeritas
