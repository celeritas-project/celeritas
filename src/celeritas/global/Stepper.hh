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
 * - \c num_track_slots : Maximum number of threads to run in parallel on GPU
 *   (optional, could be set by params)
 *   \c stream_id : Unique (thread/task) ID for this process
 * - \c action_times : Whether to synchronize device between actions for timing
 */
struct StepperInput
{
    std::shared_ptr<CoreParams const> params;
    std::shared_ptr<ActionSequence> actions;
    StreamId stream_id{};
    size_type num_track_slots{};

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
 * \note This class and its daughter may be removed soon to facilitate step
 * gathering.
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

    // Launch transport of existing states
    virtual void launch() = 0;

    //! Whether a launched step is awaiting completion
    virtual bool in_flight() const = 0;

    // Whether the launched step has completed
    virtual bool ready() const = 0;

    // Wait for and return the launched step result
    virtual StepperResult complete() = 0;

    // Transport existing states
    virtual StepperResult operator()() = 0;

    // Transport existing states and these new primaries
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
 * Device steps have separate launch and completion phases. The launch enqueues
 * the action sequence, a counter snapshot, and a completion event on the state
 * stream. Diagnostic action or step timing can still synchronize the stream.
 * Other synchronization within the action sequence is being removed
 * separately.
 *
 * \code
   Stepper<MemSpace::host> step(input);

   // Transport primaries for the initial step
   StepperResult alive_tracks = step(my_primaries);
   while (alive_tracks)
   {
       // Transport secondaries
       alive_tracks = step();
   }
   \endcode
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

    // Launch transport of existing states
    void launch() final;

    //! Whether a launched step is awaiting completion
    bool in_flight() const final { return step_in_flight_; }

    // Whether the launched step has completed
    bool ready() const final;

    // Wait for and return the launched step result
    StepperResult complete() final;

    // Transport existing states
    StepperResult operator()() final;

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
    // Preallocated result from the most recently launched step
    CounterStorage result_counters_;
    // Completion of device work and the result-counter copy
    DeviceEvent step_done_{nullptr};
    // Whether a step result is awaiting completion
    bool step_in_flight_{false};
};

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//

extern template class Stepper<MemSpace::host>;
extern template class Stepper<MemSpace::device>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
