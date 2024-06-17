//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
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
#include "celeritas/phys/Primary.hh"
#include "celeritas/track/CoreStateCounters.hh"

#include "CoreTrackData.hh"
#include "detail/CoreStateThreadOffsets.hh"

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
 *
 * When the state lives on the device, we maintain a separate copy of the
 * device "ref" in device memory: otherwise we'd have to copy the entire state
 * in launch arguments and access it through constant memory.
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

    //! Whether the state is being transported with no active particles
    bool warming_up() const
    {
        return counters_.num_active == 0 && counters_.num_primaries == 0;
    }

    //// CORE DATA ////

    //! Get a reference to the mutable state data
    Ref& ref() { return states_.ref(); }

    //! Get a reference to the mutable state data
    Ref const& ref() const { return states_.ref(); }

    //! Get a native-memspace pointer to the mutable state data
    Ptr ptr() { return ptr_; }

    //// COUNTERS ////

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

    //// ACTIONS ////

    // Resize action threads if sorting track slots
    void num_actions(size_type n);

    // Return the number of actions
    size_type num_actions() const;

    // Get a range of sorted track slots about to undergo a given action
    Range<ThreadId> get_action_range(ActionId action_id) const;

    // Access the range of actions to apply for all track IDs
    inline auto& action_thread_offsets();

    // Access the range of actions to apply for all track IDs
    inline auto const& action_thread_offsets() const;

    // Access action offsets for computation (native memory space)
    inline auto& native_action_thread_offsets();

  private:
    // State data
    CollectionStateStore<CoreStateData, M> states_;

    // Indices of first thread assigned to a given action
    detail::CoreStateThreadOffsets<M> offsets_;

    // Primaries to be added
    Collection<Primary, Ownership::value, M> primaries_;

    // Copy of state ref in device memory, if M == MemSpace::device
    DeviceVector<Ref> device_ref_vec_;

    // Native pointer to ref or
    Ptr ptr_;

    // Counters for track initialization and activity
    CoreStateCounters counters_;
};

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
/*!
 * Access the range of actions to apply for all track IDs.
 *
 * The result is size \c num_actions .
 */
template<MemSpace M>
auto& CoreState<M>::action_thread_offsets()
{
    return offsets_.host_action_thread_offsets();
}

//---------------------------------------------------------------------------//
/*!
 * Access the range of actions to apply for all track IDs.
 *
 * The result is size \c num_actions .
 */
template<MemSpace M>
auto const& CoreState<M>::action_thread_offsets() const
{
    return offsets_.host_action_thread_offsets();
}

//---------------------------------------------------------------------------//
/*!
 * Access action offsets for computation (native memory space).
 */
template<MemSpace M>
auto& CoreState<M>::native_action_thread_offsets()
{
    return offsets_.native_action_thread_offsets();
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
