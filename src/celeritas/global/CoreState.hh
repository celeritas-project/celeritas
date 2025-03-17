//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/CoreState.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/Types.hh"
#include "corecel/data/AuxInterface.hh"
#include "corecel/data/AuxStateData.hh"
#include "corecel/data/AuxStateVec.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionStateStore.hh"
#include "corecel/data/DeviceVector.hh"
#include "corecel/data/Ref.hh"
#include "corecel/sys/ThreadId.hh"
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
    //!@}

  public:
    // Support polymorphic deletion
    virtual ~CoreStateInterface();

    //! Thread/stream ID
    virtual StreamId stream_id() const = 0;

    //! Number of track slots
    virtual size_type size() const = 0;

    //! Access track initialization counters
    virtual CoreStateCounters const& counters() const = 0;

    //! Access auxiliary state data
    virtual AuxStateVec const& aux() const = 0;

  protected:
    CoreStateInterface() = default;
    CELER_DEFAULT_COPY_MOVE(CoreStateInterface);
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
    template<template<Ownership, MemSpace> class S>
    using StateRef = S<Ownership::reference, M>;

    using Ref = StateRef<CoreStateData>;
    using Ptr = ObserverPtr<Ref, M>;
    //!@}

    //! Memory space
    static constexpr MemSpace memspace = M;

  public:
    // Construct from CoreParams
    CoreState(CoreParams const& params, StreamId stream_id);

    // Construct with manual slot count
    CoreState(CoreParams const& params,
              StreamId stream_id,
              size_type num_track_slots);

    // Default destructor
    ~CoreState() final;

    // Prevent move/copy
    CELER_DELETE_COPY_MOVE(CoreState);

    //! Thread/stream ID
    StreamId stream_id() const final { return this->ref().stream_id; }

    //! Number of track slots
    size_type size() const final { return states_.size(); }

    // Set a warmup flag
    void warming_up(bool);

    //! Whether the state is being transported with no active particles
    bool warming_up() const { return warming_up_; }

    //// CORE DATA ////

    //! Get a reference to the mutable state data
    Ref& ref() { return states_.ref(); }

    //! Get a reference to the mutable state data
    Ref const& ref() const { return states_.ref(); }

    //! Get a native-memspace pointer to the mutable state data
    Ptr ptr() { return ptr_; }

    //! Reset the state data
    void reset();

    //// COUNTERS ////

    //! Track initialization counters
    CoreStateCounters& counters() { return counters_; }

    //! Track initialization counters
    CoreStateCounters const& counters() const final { return counters_; }

    //// USER DATA ////

    //! Access auxiliary state data
    AuxStateVec const& aux() const final { return aux_state_; }

    //! Access auxiliary state data (mutable)
    AuxStateVec& aux() { return aux_state_; }

    // Convenience function to access auxiliary "collection group" data
    template<template<Ownership, MemSpace> class S>
    inline StateRef<S>& aux_data(AuxId auxid);

    //// TRACK SORTING ////

    //! Return whether tracks can be sorted by action
    bool has_action_range() const { return !offsets_.empty(); }

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

    // Copy of state ref in device memory, if M == MemSpace::device
    DeviceVector<Ref> device_ref_vec_;

    // Native pointer to ref data
    Ptr ptr_;

    // Counters for track initialization and activity
    CoreStateCounters counters_;

    // User-added data associated with params
    AuxStateVec aux_state_;

    // Indices of first thread assigned to a given action
    detail::CoreStateThreadOffsets<M> offsets_;

    // Whether no primaries should be generated
    bool warming_up_{false};
};

//---------------------------------------------------------------------------//
/*!
 * Convenience function to access auxiliary "collection group" data.
 */
template<MemSpace M>
template<template<Ownership, MemSpace> class S>
auto CoreState<M>::aux_data(AuxId auxid) -> StateRef<S>&
{
    CELER_EXPECT(auxid < aux_state_.size());

    // TODO: use "checked static cast" for better runtime performance
    auto* state = dynamic_cast<AuxStateData<S, M>*>(&aux_state_.at(auxid));
    CELER_ASSERT(state);

    CELER_ENSURE(*state);
    return state->ref();
}

//---------------------------------------------------------------------------//
/*!
 * Access the range of actions to apply for all track IDs.
 */
template<MemSpace M>
auto& CoreState<M>::action_thread_offsets()
{
    return offsets_.host_action_thread_offsets();
}

//---------------------------------------------------------------------------//
/*!
 * Access the range of actions to apply for all track IDs.
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
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//

extern template class CoreState<MemSpace::host>;
extern template class CoreState<MemSpace::device>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
