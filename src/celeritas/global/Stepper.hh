//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/Stepper.hh
//! \sa Stepper.test.cc
//---------------------------------------------------------------------------//
#pragma once

#include <array>
#include <memory>
#include <vector>

#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"
#include "corecel/data/PinnedAllocator.hh"
#include "corecel/data/StateDataStore.hh"
#include "corecel/random/params/RngParamsFwd.hh"
#include "corecel/sys/DeviceEvent.hh"
#include "celeritas/Types.hh"
#include "celeritas/geo/GeoFwd.hh"
#include "celeritas/phys/Primary.hh"
#include "celeritas/track/TrackInitData.hh"
#include "celeritas/user/ActionTimes.hh"

#include "ActionSequence.hh"
#include "CoreState.hh"
#include "CoreTrackData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class CoreParams;
struct Primary;
class ExtendFromPrimariesAction;

//---------------------------------------------------------------------------//
/*!
 * State-specific options for the stepper.
 *
 * - \c params : Problem definition
 * - \c actions : Ordered sequence of transport actions
 * - \c stream_id : Unique thread or task ID for this state
 * - \c num_track_slots : Maximum number of tracks transported in parallel
 *   (optional, may be set by \c params)
 * - \c primary_capacity : Maximum primaries in either owned host buffer
 *   (optional, may be set by \c params)
 */
struct StepperInput
{
    std::shared_ptr<CoreParams const> params;
    std::shared_ptr<ActionSequence> actions;
    StreamId stream_id{};
    size_type num_track_slots{};
    size_type primary_capacity{};

    //! True if defined
    explicit operator bool() const { return params && actions && stream_id; }
};

//---------------------------------------------------------------------------//
/*!
 * Track counters for a step.
 */
struct StepperResult
{
    size_type generated{};  //!< New primaries added
    size_type queued{};  //!< Pending track initializers at end of step
    size_type active{};  //!< Active tracks at start of step
    size_type alive{};  //!< Active and alive at end of step
    size_type cut{};  //!< Tracks killed by tracking cuts during step
    size_type errored{};  //!< Tracks killed due to errors during step

    //! True if more steps need to be run
    explicit operator bool() const { return queued > 0 || alive > 0; }
};

//---------------------------------------------------------------------------//
/*!
 * Interface class for stepper classes.
 *
 * This allows higher-level classes not to care whether the stepper operates on
 * host or device.
 *
 * A stepper initially has no asynchronous result, so \c valid returns false.
 * Calling \c async, with or without primaries, starts one step and makes the
 * result valid. No other step can be started until \c get returns the result
 * and restores the initial state. The \c ready function queries completion
 * without blocking, \c wait blocks without consuming the result, and \c get
 * waits if necessary before consuming it. All three require a valid result.
 * A valid result can be ready: \c valid describes whether the result can be
 * retrieved, rather than whether device execution is still underway.
 *
 * Primary input is accumulated in a fixed-capacity producer buffer owned by
 * the stepper. Calling \c stage_primaries transfers that batch into the core
 * state and makes it available to the next call to \c async. The producer can
 * then accept another batch, including while a previous step result is valid.
 * At most one unsubmitted batch can be staged. The span overload copies its
 * input into stepper-owned storage, so the caller's span need not remain valid
 * after the call returns. The span returned by \c staged_primaries represents
 * only the current unsubmitted batch and should not be retained after a call
 * that changes the staging state.
 *
 * Calling \c async submits staged primaries, if present, and otherwise
 * advances existing tracks without changing the producer buffer. This allows
 * prior transport to be drained while a later primary batch remains buffered.
 * Primaries can be pushed and staged while a previous result is valid, but the
 * next step cannot start until \c get consumes that result. Thus result
 * completion and primary production have independent lifecycles.
 *
 * Host steps execute synchronously and are immediately ready. The deprecated
 * call operators preserve synchronous behavior by calling \c async followed
 * by \c get.
 *
 * Before destroying a device Stepper, the caller must ensure that any valid
 * asynchronous operation has completed by calling \c wait or \c get, and that
 * any staged primary batch has been submitted with \c async. Destruction does
 * not add hidden synchronization.
 *
 * \note This interface and its implementations may be removed soon to
 * facilitate step gathering.
 */
class StepperInterface
{
  public:
    //!@{
    //! \name Type aliases
    using Input = StepperInput;
    using SpanConstPrimary = Span<Primary const>;
    using result_type = StepperResult;
    using SPState = std::shared_ptr<CoreStateInterface>;
    //!@}

  public:
    // Default virtual destructor
    virtual ~StepperInterface();

    // Warm up before stepping
    virtual void warm_up() = 0;

    // Start a step with existing states and any staged primary batch
    virtual void async() = 0;

    // Copy new primaries into owned storage and start a step with them
    virtual void async(SpanConstPrimary primaries) = 0;

    //! Whether an asynchronous step result can be retrieved
    virtual bool valid() const noexcept = 0;

    // Whether the asynchronous step has completed
    virtual bool ready() const = 0;

    // Wait for the asynchronous step to complete
    virtual void wait() const = 0;

    // Wait for and return the asynchronous step result
    virtual StepperResult get() = 0;

    //! Fixed capacity of each stepper-owned primary buffer
    virtual size_type primary_capacity() const noexcept = 0;

    //! Fixed capacity of the track initializer queue
    virtual size_type initializer_capacity() const noexcept = 0;

    //! Number of primaries accumulated in the producer buffer
    virtual size_type num_buffered_primaries() const noexcept = 0;

    // Add a primary to the producer buffer; a prior result may remain valid
    virtual void push_primary(Primary primary) = 0;

    // Stage the nonempty producer buffer for the next step
    virtual void stage_primaries() = 0;

    //! Access the unsubmitted primary batch staged for the next step
    virtual SpanConstPrimary staged_primaries() const noexcept = 0;

    //! \deprecated Transport existing states
    virtual StepperResult operator()() = 0;

    // Copy primaries into owned storage and stage them for the next step
    virtual void stage_primaries(SpanConstPrimary primaries) = 0;

    // Transport existing states and these new primaries
    //! \deprecated Transport existing states and these new primaries
    virtual StepperResult operator()(SpanConstPrimary primaries) = 0;

    // Kill all tracks in flight to debug "stuck" tracks
    virtual void kill_active() = 0;

    // Reseed the RNGs at the start of an event for reproducibility
    virtual void reseed(UniqueEventId event_id) = 0;

    //! Get action sequence for timing diagnostics
    virtual ActionSequence const& actions() const = 0;

    //! Get the core state interface
    virtual CoreStateInterface const& state() const = 0;

    //! Get a shared pointer to the state (TEMPORARY)
    virtual SPState sp_state() = 0;

  protected:
    StepperInterface() = default;
    CELER_DEFAULT_COPY_MOVE(StepperInterface);
};

//---------------------------------------------------------------------------//
/*!
 * Manage a state vector and execute a single step on all of them.
 *
 * \note This is likely to be removed and refactored since we're changing how
 * primaries are created and how multithread state ownership is managed.
 *
 * Device steps have separate start and result phases. Calling \c async
 * enqueues the action sequence, a counter snapshot, and a completion event on
 * the state stream. Diagnostic action or step timing can still synchronize the
 * stream. Other synchronization within the action sequence is being removed
 * separately.
 *
 * The two primary buffers are reserved to \c primary_capacity at construction,
 * so accumulating and staging primaries within capacity does not reallocate.
 * Device buffers use pinned host memory so inserting a staged batch can enqueue
 * its host-to-device copy. The submitted source storage is retained until that
 * copy completes; same-stream ordering ensures the step actions see the copied
 * input. Reusing a submitted source may wait for its copy event, but does not
 * wait for the full step to complete.
 *
 * The following sequence overlaps production of a later primary batch with an
 * outstanding result:
 * \code
   Stepper<MemSpace::device> step(input);

   step.async(make_span(first_primaries));
   for (Primary primary : second_primaries)
   {
       step.push_primary(std::move(primary));
   }
   StepperResult result = step.get();
   while (result)
   {
       step.async();
       result = step.get();
   }

   step.stage_primaries();
   step.async();
   StepperResult second_result = step.get();
   \endcode
 *
 * \internal
 *
 * \par Asynchronous state
 *
 * The \c valid_ flag tracks whether a result can be consumed, whereas
 * \c step_done_ tracks completion of device work. Their states after successful
 * calls are:
 *
 * | Lifecycle point | \c valid_ | CPU \c step_done_ | GPU \c step_done_ |
 * | --------------- | -------------- | ------------------ | ------------------ |
 * | Construction or after \c get | false | Null | Allocated and ready |
 * | After \c async | true | Null and ready | Recorded; pending or ready |
 * | After \c ready returns false | true | Not possible | Recorded and pending |
 * | After \c ready is true or \c wait | true | Null/ready | Recorded/ready |
 *
 * A host step executes synchronously, so its null event is always ready. A
 * device event is allocated once and re-recorded after each counter snapshot.
 * Calling \c get first waits for completion and then clears \c valid_;
 * it does not reset or replace the event. Primary buffering has an independent
 * state, tracked by \c primary_phase_ and \c primary_copy_done_. The staged
 * storage remains internal while a copy source is submitted, but the public
 * \c staged_primaries accessor returns only an unsubmitted batch:
 *
 * | Primary phase | Producer | Staged storage | Accessor | Copy event |
 * | ------------- | -------- | -------------- | -------- | ---------- |
 * | \c empty | May fill | Empty | Empty | Null/ready or allocated/ready |
 * | \c staged | May fill | Next input | Next input | Recorded after H2D copy |
 * | \c submitted | May fill | Prior copy source | Empty | Pending or ready |
 *
 * Calling \c stage_primaries changes \c empty to \c staged. Calling \c async
 * changes \c staged to \c submitted after the actions have been enqueued. A
 * later \c stage_primaries call may reclaim a submitted source after waiting
 * only for its copy event, then stage the producer buffer. Calling \c get also
 * reclaims a submitted source, but leaves a next \c staged batch unchanged.
 * Host copies complete synchronously, so \c primary_copy_done_ is null and
 * reclaiming the source does not wait.
 *
 * The expected state transitions are
 * \code
 *   no result + producer -- async() --> valid result + same producer
 *   no result + no queued input -- async(primaries) --> valid result
 *   valid result -- ready() or wait() --> valid result
 *   valid result -- get() --> no result
 *
 *   producer -- stage_primaries() --> staged
 *   staged + no result -- async() --> submitted + valid result
 *   submitted + producer -- stage_primaries() --> staged
 * \endcode
 * Calling \c ready or \c wait repeatedly with a valid result is allowed. The
 * next \c async call is allowed only after \c get consumes the previous result.
 * Primaries may be pushed and staged while a result is valid, and the producer
 * may begin filling again while that next batch is staged. The staged batch
 * cannot be submitted until the prior result is consumed. Calls to \c warm_up,
 * \c reset_state, and \c reseed are rejected while a result or queued primary
 * batch exists. Calling \c kill_active permits buffered primaries but rejects a
 * pending result or staged batch. The synchronous call operators perform \c
 * async followed immediately by \c get.
 */
template<MemSpace M>
class Stepper final : public StepperInterface
{
  public:
    //!@{
    //! \name Type aliases
    using StateRef = CoreStateData<Ownership::reference, M>;
    //!@}

  public:
    // Construct with problem parameters and setup options
    explicit Stepper(Input input);

    // Default destructor
    ~Stepper() final;

    // Warm up before stepping
    void warm_up() final;

    // Start a step with existing states and any staged primary batch
    void async() final;

    // Copy new primaries into owned storage and start a step with them
    void async(SpanConstPrimary primaries) final;

    //! Whether an asynchronous step result can be retrieved
    bool valid() const noexcept final { return valid_; }

    // Whether the asynchronous step has completed
    bool ready() const final;

    // Wait for the asynchronous step to complete
    void wait() const final;

    // Wait for and return the asynchronous step result
    StepperResult get() final;

    //! Fixed capacity of each stepper-owned primary buffer
    size_type primary_capacity() const noexcept final
    {
        return primary_capacity_;
    }

    // Fixed capacity of the track initializer queue
    size_type initializer_capacity() const noexcept final;

    //! Number of primaries accumulated in the producer buffer
    size_type num_buffered_primaries() const noexcept final
    {
        return primary_buffer_.size();
    }

    // Add a primary to the producer buffer; a prior result may remain valid
    void push_primary(Primary primary) final;

    // Stage the nonempty producer buffer for the next step
    void stage_primaries() final;

    //! Access the unsubmitted primary batch staged for the next step
    SpanConstPrimary staged_primaries() const noexcept final;

    // Transport existing states
    StepperResult operator()() final;

    // Copy primaries into owned storage and stage them for the next step
    void stage_primaries(SpanConstPrimary primaries) final;

    // Transport existing states and these new primaries
    StepperResult operator()(SpanConstPrimary primaries) final;

    // Kill all tracks in flight to debug "stuck" tracks
    void kill_active() final;

    // Reseed the RNGs at the start of an event for reproducibility
    void reseed(UniqueEventId event_id) final;

    //! Get action sequence for timing diagnostics
    ActionSequence const& actions() const final { return *actions_; }

    //! Access core data, primarily for debugging
    StateRef const& state_ref() const { return state_->ref(); }

    //! Get the core state interface for diagnostic output
    CoreStateInterface const& state() const final { return *state_; }

    // Reset the core state counters and data so it can be reused
    void reset_state();

    //! Get a shared pointer to the state (TEMPORARY, DO NOT USE)
    SPState sp_state() final { return state_; }

  private:
    enum class PrimaryPhase
    {
        empty,
        staged,
        submitted
    };

    using VecPrimary = std::vector<Primary>;
    using PinnedVecPrimary = std::vector<Primary, PinnedAllocator<Primary>>;
    using PrimaryStorage = MemSpaceCond_t<M, VecPrimary, PinnedVecPrimary>;
    using PinnedVecCounters
        = std::vector<CoreStateCounters, PinnedAllocator<CoreStateCounters>>;
    using CounterStorage
        = MemSpaceCond_t<M, std::array<CoreStateCounters, 1>, PinnedVecCounters>;

    // Params data
    std::shared_ptr<CoreParams const> params_;
    // Call sequence
    std::shared_ptr<ActionSequence> actions_;
    // Primary initialization
    std::shared_ptr<ExtendFromPrimariesAction const> primaries_action_;
    // State data
    std::shared_ptr<CoreState<M>> state_;
    // Maximum number of primaries in each host buffer
    size_type primary_capacity_{};
    // Primaries being accumulated by the producer
    PrimaryStorage primary_buffer_;
    // Host source for a staged or submitted primary copy
    PrimaryStorage staged_primaries_;
    // Completion of the staged-primary H2D copy
    DeviceEvent primary_copy_done_{nullptr};
    // Logical state of staged_primaries_
    PrimaryPhase primary_phase_{PrimaryPhase::empty};
    // Preallocated result from the most recently started step
    CounterStorage result_counters_;
    // Completion of device work and the result-counter copy
    DeviceEvent step_done_{nullptr};
    // Whether an asynchronous step result can be retrieved
    bool valid_{false};

    // Whether an operation would conflict with queued primaries
    bool has_queued_primaries() const noexcept;

    // Release a submitted primary source after its copy completes
    void reclaim_submitted_primaries();
};

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//

extern template class Stepper<MemSpace::host>;
extern template class Stepper<MemSpace::device>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
