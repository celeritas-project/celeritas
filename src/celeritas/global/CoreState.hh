//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/CoreState.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionStateStore.hh"
#include "corecel/data/DeviceVector.hh"
#include "corecel/data/Ref.hh"
#include "corecel/sys/ThreadId.hh"
#include "celeritas/global/CoreTrackData.hh"
#include "celeritas/phys/Primary.hh"
#include "celeritas/track/CoreStateCounters.hh"

namespace celeritas
{
class CoreParams;
//---------------------------------------------------------------------------//
/*!
 * Abstract base class for CoreState.
 */
class CoreStateInterface
{
  public:
    //!@{
    //! \name Type aliases
    using size_type = TrackSlotId::size_type;
    using PrimaryRange = ItemRange<Primary>;
    //!@}

  public:
    //! Thread/stream ID
    virtual StreamId stream_id() const = 0;

    //! Number of track slots
    virtual size_type size() const = 0;

    //! Access track initialization counters
    virtual CoreStateCounters const& counters() const = 0;

    // Inject primaries to be turned into TrackInitializers
    virtual void insert_primaries(Span<Primary const> host_primaries) = 0;

  protected:
    ~CoreStateInterface() = default;
};

//---------------------------------------------------------------------------//
/*!
 * Store all state data for a single thread.
 */
template<MemSpace M>
class CoreState final : public CoreStateInterface
{
  public:
    //!@{
    //! \name Type aliases
    using Ref = CoreStateData<Ownership::reference, M>;
    using Ptr = ObserverPtr<Ref, M>;
    using PrimaryCRef = Collection<Primary, Ownership::const_reference, M>;
    template<MemSpace M2>
    using ThreadItems = Collection<ThreadId, Ownership::value, M2, ActionId>;
    //!@}

  public:
    // Construct from CoreParams
    CoreState(CoreParams const& params,
              StreamId stream_id,
              size_type num_track_slots);

    //! Thread/stream ID
    StreamId stream_id() const final { return this->ref().stream_id; }

    //! Number of track slots
    size_type size() const final { return states_.size(); }

    //! Get a reference to the mutable state data
    Ref& ref() { return states_.ref(); }

    //! Get a reference to the mutable state data
    Ref const& ref() const { return states_.ref(); }

    // Get a native-memspace pointer to the mutable state data
    inline Ptr ptr();

    //! Track initialization counters
    CoreStateCounters& counters() { return counters_; }

    //! Track initialization counters
    CoreStateCounters const& counters() const final { return counters_; }

    //// PRIMARY STORAGE ////

    // Inject primaries to be turned into TrackInitializers
    void insert_primaries(Span<Primary const> host_primaries) final;

    // Get the range of valid primaries
    inline PrimaryRange primary_range() const;

    // Get the storage for primaries
    inline PrimaryCRef primary_storage() const;

    //! Clear primaries after constructing initializers from them
    void clear_primaries() { counters_.num_primaries = 0; }

    void resize_offsets(size_type n);

    void count_tracks_per_action(TrackOrder order);

    //! Get a range delimiting the [start, end) of the track partition assigned
    //! action_id in track_slots
    Range<ThreadId> get_action_range(ActionId action_id) const;

    ThreadItems<MemSpace::host>& host_thread_offsets();

  private:
    // State data
    CollectionStateStore<CoreStateData, M> states_;

    //
    ThreadItems<M> thread_offsets_;
    ThreadItems<MemSpace::host> host_thread_offsets_;

    // Primaries to be added
    Collection<Primary, Ownership::value, M> primaries_;

    // Copy of state ref in device memory, if M == MemSpace::device
    DeviceVector<Ref> device_ref_vec_;

    // Counters for track initialization and activity
    CoreStateCounters counters_;
};

//---------------------------------------------------------------------------//
/*!
 * Access a native pointer to a NativeCRef.
 *
 * This way, CUDA kernels only need to copy a pointer in the kernel arguments,
 * rather than the entire (rather large) DeviceRef object.
 */
template<MemSpace M>
auto CoreState<M>::ptr() -> Ptr
{
    if constexpr (M == MemSpace::host)
    {
        return make_observer(&this->ref());
    }
    else if constexpr (M == MemSpace::device)
    {
        CELER_ENSURE(!device_ref_vec_.empty());
        return make_observer(device_ref_vec_);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Get the range of valid primaries.
 */
template<MemSpace M>
auto CoreState<M>::primary_range() const -> PrimaryRange
{
    return {ItemId<Primary>(counters_.num_primaries)};
}

//---------------------------------------------------------------------------//
/*!
 * Get the range of valid primaries.
 */
template<MemSpace M>
auto CoreState<M>::primary_storage() const -> PrimaryCRef
{
    return PrimaryCRef{primaries_};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
