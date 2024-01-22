//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/CoreState.cc
//---------------------------------------------------------------------------//
#include "CoreState.hh"

#include "corecel/data/Copier.hh"
#include "corecel/io/Logger.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "celeritas/track/detail/TrackSortUtils.hh"

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

    ScopedProfiling profile_this{"construct-state"};

    states_ = CollectionStateStore<CoreStateData, M>(
        params.host_ref(), stream_id, num_track_slots);

    counters_.num_vacancies = num_track_slots;
    counters_.num_primaries = 0;
    counters_.num_initializers = 0;

    if constexpr (M == MemSpace::device)
    {
        device_ref_vec_ = DeviceVector<Ref>(1);
        device_ref_vec_.copy_to_device({&this->ref(), 1});
        ptr_ = make_observer(device_ref_vec_);
    }
    else if constexpr (M == MemSpace::host)
    {
        ptr_ = make_observer(&this->ref());
    }

    CELER_LOG_LOCAL(status) << "Celeritas core state initialization complete";
    CELER_ENSURE(states_);
    CELER_ENSURE(ptr_);
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

//---------------------------------------------------------------------------//
/*!
 * Get a range delimiting the [start, end) of the track partition assigned
 * action_id in track_slots
 */
template<MemSpace M>
Range<ThreadId> CoreState<M>::get_action_range(ActionId action_id) const
{
    auto const& thread_offsets = offsets_.host_action_thread_offsets();
    CELER_EXPECT((action_id + 1) < thread_offsets.size());
    return {thread_offsets[action_id], thread_offsets[action_id + 1]};
}

//---------------------------------------------------------------------------//
/*!
 * resize ActionThreads collection to the number of actions
 */
template<MemSpace M>
void CoreState<M>::num_actions(size_type n)
{
    offsets_.resize(n);
}

//---------------------------------------------------------------------------//
/*!
 * Return the number of actions, i.e. thread_offsets_ size
 */
template<MemSpace M>
size_type CoreState<M>::num_actions() const
{
    return offsets_.host_action_thread_offsets().size();
}

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//
template class CoreState<MemSpace::host>;
template class CoreState<MemSpace::device>;
//---------------------------------------------------------------------------//
}  // namespace celeritas
