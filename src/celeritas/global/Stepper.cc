//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/Stepper.cc
//---------------------------------------------------------------------------//
#include "Stepper.hh"

#include <utility>

#include "corecel/Assert.hh"
#include "corecel/data/Copier.hh"
#include "corecel/data/Ref.hh"
#include "corecel/random/params/RngParams.hh"
#include "corecel/sys/ActionRegistry.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "corecel/sys/Stream.hh"
#include "orange/OrangeData.hh"
#include "celeritas/Types.hh"
#include "celeritas/random/RngReseed.hh"
#include "celeritas/track/ExtendFromPrimariesAction.hh"
#include "celeritas/track/TrackInitParams.hh"

#include "CoreParams.hh"

#include "detail/KillActive.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Call a function when this object is destroyed (at end of scope).
 */
template<class F>
class ScopeExit
{
  public:
    //! Construct with functor
    ScopeExit(F func) : func_{std::forward<F>(func)} {}

    //! Call functor on destruction
    ~ScopeExit() { func_(); }

    CELER_DELETE_COPY_MOVE(ScopeExit);

  private:
    F func_;
};

template<class F>
ScopeExit(F&& func) -> ScopeExit<F>;

//---------------------------------------------------------------------------//
//! Convert internal state counters to a public step result
StepperResult make_stepper_result(CoreStateCounters const& counters)
{
    StepperResult result;
    result.generated = counters.num_generated;
    result.active = counters.num_active;
    result.alive = counters.num_alive;
    result.queued = counters.num_initializers;
    result.cut = counters.num_cut;
    result.errored = counters.num_errored;
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
StepperInterface::~StepperInterface() = default;

//---------------------------------------------------------------------------//
/*!
 * Construct with problem parameters and setup options.
 */
template<MemSpace M>
Stepper<M>::Stepper(Input input)
    : params_(std::move(input.params)), actions_(std::move(input.actions))
{
    CELER_EXPECT(params_);
    CELER_EXPECT(actions_);

    // Save primary action: TODO this is a hack and should be refactored so
    // that we pass generators into the stepper and eliminate the call
    // signature with primaries
    primaries_action_ = ExtendFromPrimariesAction::find_action(*params_);
    CELER_VALIDATE(primaries_action_,
                   << "primary generator was not added to the stepping loop");

    // An override lets callers size both buffers for their batching policy
    // while preserving the configured per-stream default for existing users.
    primary_capacity_ = input.primary_capacity;
    if (primary_capacity_ == 0)
    {
        primary_capacity_ = params_->sizes().primaries
                            / params_->sizes().streams;
    }
    CELER_VALIDATE(
        primary_capacity_ > 0,
        << "primary capacity is smaller than the number of streams");
    primary_buffer_.reserve(primary_capacity_);
    staged_primaries_.reserve(primary_capacity_);

    size_type const track_slots = (input.num_track_slots == 0
                                       ? params_->tracks_per_stream()
                                       : input.num_track_slots);
    CELER_VALIDATE(track_slots > 0,
                   << "track slots were specified neither in core params nor "
                      "stepper input");
    // Create state, including aux data
    state_ = std::make_shared<CoreState<M>>(
        *params_, input.stream_id, track_slots);

    if constexpr (M == MemSpace::device)
    {
        // Allocate reusable asynchronous state before stepping begins
        result_counters_.resize(1);
        primary_copy_done_ = DeviceEvent{celeritas::device()};
        step_done_ = DeviceEvent{celeritas::device()};
    }

    // Execute beginning-of-run action
    ScopedProfiling profile_this{"begin-run"};
    actions_->begin_run(*params_, *state_);
    CELER_ENSURE(result_counters_.size() == 1);
}

//---------------------------------------------------------------------------//
//! Default destructor
template<MemSpace M>
Stepper<M>::~Stepper() = default;

//---------------------------------------------------------------------------//
/*!
 * Run all step actions with no active particles.
 *
 * The warmup stage is useful for profiling and debugging since the first
 * step iteration can do the following:
 * - Initialize asynchronous memory pools
 * - Interrogate kernel functions for properties to be output later
 * - Allocate "lazy" auxiliary data (e.g. action diagnostics)
 */
template<MemSpace M>
void Stepper<M>::warm_up()
{
    CELER_VALIDATE(!valid_, << "cannot warm up with a pending step");
    CELER_VALIDATE(!this->has_queued_primaries(),
                   << "cannot warm up with queued primaries");
    CELER_VALIDATE(state_->sync_get_counters().num_active == 0,
                   << "cannot warm up when state has active tracks");

    ScopedProfiling profile_this{"warmup"};
    state_->warming_up(true);
    ScopeExit on_exit_{[this] { state_->warming_up(false); }};
    actions_->step(*params_, *state_);
    CELER_ENSURE(state_->sync_get_counters().num_active == 0);
}

//---------------------------------------------------------------------------//
/*!
 * Start a step with existing states and any staged primary batch.
 *
 * A single transport step is simply a loop over a topologically sorted DAG
 * of kernels. The step result must be retrieved with \c get before another
 * step can be started. In device mode the result counters are copied
 * asynchronously to pinned host memory, followed by a completion event.
 * Primaries in the producer buffer remain there unless they have first been
 * staged.
 *
 * Existing synchronization within the action sequence can still block this
 * call. Removing those counter-dependent synchronization points is handled
 * separately.
 */
template<MemSpace M>
void Stepper<M>::async()
{
    CELER_VALIDATE(
        !valid_,
        << "cannot start a step before the current step has been consumed");

    ScopedProfiling profile_this{"step"};
    auto counters = state_->sync_get_counters();
    counters.num_generated = 0;
    counters.num_cut = 0;
    counters.num_errored = 0;
    state_->sync_put_counters(counters);
    actions_->step(*params_, *state_);
    if (primary_phase_ == PrimaryPhase::staged)
    {
        // The action sequence has enqueued work that consumes the staged input,
        // but its host source remains protected until the H2D copy completes.
        primary_phase_ = PrimaryPhase::submitted;
    }

    if constexpr (M == MemSpace::device)
    {
        auto const* counters_ptr = static_cast<CoreStateCounters const*>(
            state_->ref().init.counters.data());
        Copier<CoreStateCounters, MemSpace::host> copy_counters{
            make_span(result_counters_), state_->stream_id()};
        copy_counters(MemSpace::device, {counters_ptr, 1});
        step_done_.record(celeritas::device().stream(state_->stream_id()));
    }
    else
    {
        result_counters_.front() = state_->sync_get_counters();
    }
    valid_ = true;
    CELER_ENSURE(primary_phase_ != PrimaryPhase::staged);
}

//---------------------------------------------------------------------------//
/*!
 * Copy new primaries into owned storage and start a step with them.
 *
 * The input is copied into the producer buffer before its host-to-device copy
 * is enqueued, so it can be released when this function returns.
 */
template<MemSpace M>
void Stepper<M>::async(SpanConstPrimary primaries)
{
    CELER_VALIDATE(
        !valid_,
        << "cannot start a step before the current step has been consumed");
    this->stage_primaries(primaries);
    this->async();
}

//---------------------------------------------------------------------------//
/*!
 * Return the fixed capacity of the track initializer queue.
 *
 * After consuming a step result, callers can compare this with the result's
 * queued initializers, the producer buffer size, and \c secondary_capacity
 * before staging primaries.
 */
template<MemSpace M>
size_type Stepper<M>::initializer_capacity() const noexcept
{
    return params_->init()->capacity();
}

//---------------------------------------------------------------------------//
/*!
 * Add a primary to the producer buffer.
 *
 * The producer buffer can be filled while a step result is valid or another
 * primary batch is staged. Its fixed capacity is reserved at construction.
 */
template<MemSpace M>
void Stepper<M>::push_primary(Primary primary)
{
    CELER_VALIDATE(primary_buffer_.size() < primary_capacity_,
                   << "primary buffer capacity of " << primary_capacity_
                   << " exceeded");
    primary_buffer_.push_back(std::move(primary));
}

//---------------------------------------------------------------------------//
/*!
 * Stage the producer buffer for transport.
 *
 * This validates and inserts primaries into the stepper state but does not
 * execute transport actions. This separation does not by itself make staging
 * nonblocking: in device mode the current counter updates may still
 * synchronize internally. Reusing the source of a previously submitted batch
 * waits only for its copy event, not for completion of the previous step.
 *
 * \pre No primaries are currently staged.
 */
template<MemSpace M>
void Stepper<M>::stage_primaries()
{
    CELER_VALIDATE(!primary_buffer_.empty(),
                   << "cannot stage an empty primary buffer");
    CELER_VALIDATE(primary_phase_ != PrimaryPhase::staged,
                   << "cannot stage primaries while another batch is staged");

    this->reclaim_submitted_primaries();

    auto primaries = make_span(primary_buffer_);
    CELER_EXPECT(!primaries.empty());
    CELER_EXPECT(primaries_action_);

    // Check that events are consistent with our 'max events'
    auto max_id
        = std::max_element(primaries.begin(),
                           primaries.end(),
                           [](Primary const& left, Primary const& right) {
                               return left.event_id < right.event_id;
                           });
    CELER_ASSERT(max_id->event_id);
    CELER_VALIDATE(max_id->event_id < params_->init()->max_events(),
                   << "event number " << max_id->event_id.unchecked_get()
                   << " exceeds max_events=" << params_->init()->max_events());

    auto counters = state_->sync_get_counters();
    CELER_VALIDATE(counters.num_pending == 0,
                   << "cannot stage " << primaries.size()
                   << " primaries while " << counters.num_pending
                   << " primaries are already pending");
    counters.num_pending = primaries.size();
    state_->sync_put_counters(counters);
    primaries_action_->insert(*params_, *state_, primaries);
    if constexpr (M == MemSpace::device)
    {
        primary_copy_done_.record(
            celeritas::device().stream(state_->stream_id()));
    }

    primary_buffer_.swap(staged_primaries_);
    primary_phase_ = PrimaryPhase::staged;
}

//---------------------------------------------------------------------------//
/*!
 * Copy user-provided primaries into owned storage and stage them.
 *
 * The caller's span can be released when this function returns. If staging
 * fails, the copied input is discarded so the caller can correct it and retry.
 */
template<MemSpace M>
void Stepper<M>::stage_primaries(SpanConstPrimary primaries)
{
    CELER_EXPECT(!primaries.empty());
    CELER_VALIDATE(primary_buffer_.empty(),
                   << "cannot stage external primaries while the primary "
                      "buffer is not empty");
    CELER_VALIDATE(primary_phase_ != PrimaryPhase::staged,
                   << "cannot stage primaries while another batch is staged");
    CELER_VALIDATE(primaries.size() <= primary_capacity_,
                   << "primary buffer capacity of " << primary_capacity_
                   << " is insufficient for " << primaries.size()
                   << " primaries");

    primary_buffer_.insert(
        primary_buffer_.end(), primaries.begin(), primaries.end());
    try
    {
        this->stage_primaries();
    }
    catch (...)
    {
        primary_buffer_.clear();
        throw;
    }
}

//---------------------------------------------------------------------------//
/*!
 * Access the unsubmitted primaries staged for the next step.
 *
 * Submitted source storage is retained internally until its copy completes,
 * but is not exposed by this accessor.
 */
template<MemSpace M>
auto Stepper<M>::staged_primaries() const noexcept -> SpanConstPrimary
{
    if (primary_phase_ != PrimaryPhase::staged)
    {
        return {};
    }
    return make_span(staged_primaries_);
}

//---------------------------------------------------------------------------//
/*!
 * Whether the asynchronous step has completed.
 */
template<MemSpace M>
bool Stepper<M>::ready() const
{
    CELER_VALIDATE(valid_, << "cannot query readiness without a pending step");
    return step_done_.ready();
}

//---------------------------------------------------------------------------//
/*!
 * Wait for the asynchronous step to complete without consuming its result.
 */
template<MemSpace M>
void Stepper<M>::wait() const
{
    CELER_VALIDATE(valid_, << "cannot wait without a pending step");
    step_done_.sync();
}

//---------------------------------------------------------------------------//
/*!
 * Wait for and return the asynchronous step result.
 *
 * Calling this consumes the pending result and allows another step to be
 * started.
 */
template<MemSpace M>
auto Stepper<M>::get() -> result_type
{
    CELER_VALIDATE(valid_, << "cannot get without a pending step");

    this->wait();
    auto result = make_stepper_result(result_counters_.front());
    valid_ = false;
    this->reclaim_submitted_primaries();
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Transport already-initialized states.
 *
 * \deprecated This is the deprecated synchronous compatibility wrapper for \c
 * async and
 * \c get.
 */
template<MemSpace M>
auto Stepper<M>::operator()() -> result_type
{
    this->async();
    return this->get();
}

//---------------------------------------------------------------------------//
/*!
 * \deprecated Initialize new primaries and transport them for a single step.
 */
template<MemSpace M>
auto Stepper<M>::operator()(SpanConstPrimary primaries) -> result_type
{
    this->async(primaries);
    return this->get();
}

//---------------------------------------------------------------------------//
/*!
 * Kill all tracks in flight to debug "stuck" tracks.
 *
 * The next "step" will apply the tracking cut and (if CPU) print diagnostic
 * output about the failed tracks. Primaries in the producer buffer are not yet
 * part of the core state and remain unchanged, but staged primaries prevent
 * this operation.
 */
template<MemSpace M>
void Stepper<M>::kill_active()
{
    CELER_VALIDATE(
        !valid_, << "cannot kill active tracks while an asynchronous step is executing");
    CELER_VALIDATE(primary_phase_ != PrimaryPhase::staged,
                   << "cannot kill active tracks with staged primaries");
    CELER_LOG_LOCAL(error) << "Killing "
                           << state_->sync_get_counters().num_active
                           << " active tracks";
    detail::kill_active(*params_, *state_);
}

//---------------------------------------------------------------------------//
/*!
 * Reseed RNGs and counters at the start of an event for reproducibility.
 *
 * This reinitializes the RNG states using a single seed and unique subsequence
 * for each thread. It ensures that given an event identification, the random
 * number sequence for the event (and thus the event's behavior) can be
 * reproduced.
 */
template<MemSpace M>
void Stepper<M>::reseed(UniqueEventId event_id)
{
    CELER_VALIDATE(!valid_,
                   << "cannot reseed while an asynchronous step is executing");
    CELER_VALIDATE(!this->has_queued_primaries(),
                   << "cannot reseed with queued primaries");
    ScopedProfiling profile_this{"reseed"};
    reseed_rng(get_ref<M>(*params_->rng()),
               state_->ref().rng,
               state_->stream_id(),
               event_id);
    params_->init()->reset_track_ids(state_->stream_id(), &state_->ref().init);
}

//---------------------------------------------------------------------------//
/*!
 * Reset the core state counters and data so it can be reused.
 */
template<MemSpace M>
void Stepper<M>::reset_state()
{
    CELER_VALIDATE(
        !valid_,
        << "cannot reset state while an asynchronous step is executing");
    CELER_VALIDATE(!this->has_queued_primaries(),
                   << "cannot reset state with queued primaries");
    state_->reset();
}

//---------------------------------------------------------------------------//
/*!
 * Whether an operation would conflict with queued primaries.
 */
template<MemSpace M>
bool Stepper<M>::has_queued_primaries() const noexcept
{
    // A submitted copy source is owned by the pending result lifecycle, which
    // callers validate separately through valid_.
    return !primary_buffer_.empty() || primary_phase_ == PrimaryPhase::staged;
}

//---------------------------------------------------------------------------//
/*!
 * Release a submitted primary source after its copy completes.
 */
template<MemSpace M>
void Stepper<M>::reclaim_submitted_primaries()
{
    if (primary_phase_ == PrimaryPhase::submitted)
    {
        // Wait only for the copy source lifetime, not for step completion.
        primary_copy_done_.sync();
        staged_primaries_.clear();
        primary_phase_ = PrimaryPhase::empty;
    }
}

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//

template class Stepper<MemSpace::host>;
template class Stepper<MemSpace::device>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
