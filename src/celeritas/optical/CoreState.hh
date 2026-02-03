//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/CoreState.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/cont/Span.hh"
#include "corecel/data/AuxInterface.hh"
#include "corecel/data/AuxStateVec.hh"
#include "corecel/data/CollectionStateStore.hh"
#include "corecel/data/ObserverPtr.hh"
#include "corecel/random/params/RngParamsFwd.hh"
#include "celeritas/Types.hh"
#include "celeritas/phys/GeneratorCounters.hh"
#include "celeritas/track/CoreStateCounters.hh"

#include "CoreTrackData.hh"
#include "TrackInitializer.hh"

namespace celeritas
{
namespace optical
{
class CoreParams;

//---------------------------------------------------------------------------//
/*!
 * Interface class for optical state data.
 *
 * This inherits from the "aux state" interface to allow stream-local storage
 * with the optical offload data.
 */
class CoreStateInterface : public AuxStateInterface
{
  public:
    //!@{
    //! \name Type aliases
    using size_type = TrackSlotId::size_type;
    //!@}

  public:
    // Support polymorphic deletion
    ~CoreStateInterface() override;

    //! Thread/stream ID
    virtual StreamId stream_id() const = 0;

    //! Synchronize and copy track initialization counters from device to host
    //! For host-only code, this replaces the old counters() function
    [[nodiscard]] virtual CoreStateCounters sync_get_counters() const = 0;

    //! Synchronize and copy track initialization counters from host to device
    //! For host-only code, this replaces the old counters() function
    //! since we return a CoreStateCounters object instead of a reference
    virtual void sync_put_counters(CoreStateCounters const&) = 0;

    //! Reseed the RNGs at the start of an event for reproducibility
    virtual void reseed(std::shared_ptr<RngParams const>, UniqueEventId) = 0;

    //! Number of track slots
    virtual size_type size() const = 0;

    // Inject optical primaries
    virtual void insert_primaries(Span<TrackInitializer const> host_primaries)
        = 0;

  protected:
    CoreStateInterface() = default;

    CELER_DEFAULT_COPY_MOVE(CoreStateInterface);
};

//---------------------------------------------------------------------------//
/*!
 * Manage the optical state counters and auxiliary data.
 */
class CoreStateBase : public CoreStateInterface
{
  public:
    //!@{
    //! \name Type aliases
    using SPAuxStateVec = std::shared_ptr<AuxStateVec>;
    //!@}

  public:
    //! Optical loop statistics
    CounterAccumStats const& accum() const { return accum_; }

    //! Optical loop statistics
    CounterAccumStats& accum() { return accum_; }

    //// AUXILIARY DATA ////

    //! Access auxiliary core state data
    SPAuxStateVec const& aux() const { return aux_state_; }

    //! Access auxiliary core state data (mutable)
    SPAuxStateVec& aux() { return aux_state_; }

  protected:
    CoreStateBase() = default;

    // Anchor vtable
    ~CoreStateBase() override;

  private:
    //! Counts accumulated over the event for diagnostics
    CounterAccumStats accum_;

    // Auxiliary data owned by the core state
    SPAuxStateVec aux_state_;
};

//---------------------------------------------------------------------------//
/*!
 * Store all state data for a single thread.
 *
 * When the state lives on the device, we maintain a separate copy of the
 * device "ref" in device memory: otherwise we'd have to copy the entire state
 * in launch arguments and access it through constant memory.
 *
 * \todo Encapsulate all the action management accessors in a helper class.
 */
template<MemSpace M>
class CoreState final : public CoreStateBase
{
  public:
    //!@{
    //! \name Type aliases
    template<template<Ownership, MemSpace> class S>
    using StateRef = S<Ownership::reference, M>;

    using Ref = StateRef<CoreStateData>;
    using Ptr = ObserverPtr<Ref, M>;
    //!@}

  public:
    // Construct from CoreParams
    CoreState(CoreParams const& params,
              StreamId stream_id,
              size_type num_track_slots);

    // Default destructor
    ~CoreState() final;

    //! Thread/stream ID
    StreamId stream_id() const final { return this->ref().stream_id; }

    //! Number of track slots
    size_type size() const final { return states_.size(); }

    //! Synchronize and copy track initialization counters from device to host
    [[nodiscard]] CoreStateCounters sync_get_counters() const final;

    //! Synchronize and copy track initialization counters from host to device
    //! For host-only code, this copies the local CoreStateCounters back to the
    //! class, since sync_get_counters() doesn't return a reference
    void sync_put_counters(CoreStateCounters const&) final;

    // Whether the state is being transported with no active particles
    bool warming_up() const;

    //// CORE DATA ////

    //! Get a reference to the mutable state data
    Ref& ref() { return states_.ref(); }

    //! Get a reference to the mutable state data
    Ref const& ref() const { return states_.ref(); }

    //! Get a native-memspace pointer to the mutable state data
    Ptr ptr() { return ptr_; }

    // Reset the data for a new step
    void reset();

    // Reseed the RNGs at the start of an event for reproducibility
    void reseed(std::shared_ptr<RngParams const>, UniqueEventId) final;

    // Inject primaries to be turned into TrackInitializers
    void insert_primaries(Span<TrackInitializer const> host_primaries) final;

  private:
    // State data
    CollectionStateStore<CoreStateData, M> states_;

    // Copy of state ref in device memory, if M == MemSpace::device
    DeviceVector<Ref> device_ref_vec_;

    // Native pointer to ref or
    Ptr ptr_;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
