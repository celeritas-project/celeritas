//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/Stepper.cc
//---------------------------------------------------------------------------//
#include "Stepper.hh"

#include <utility>

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
        step_done_ = DeviceEvent{celeritas::device()};
    }

    // Execute beginning-of-run action
    ScopedProfiling profile_this{"begin-run"};
    actions_->begin_run(*params_, *state_);
}

//---------------------------------------------------------------------------//
/*!
 * Synchronize outstanding device work before releasing its state.
 */
template<MemSpace M>
Stepper<M>::~Stepper()
{
    if constexpr (M == MemSpace::device)
    {
        try
        {
            // Include work that may follow the step-completion event
            celeritas::device().stream(state_->stream_id()).sync();
        }
        catch (RuntimeError const& e)
        {
            CELER_LOG_LOCAL(error)
                << "Failed to synchronize Stepper during destruction: "
                << e.what();
        }
        catch (...)
        {
            CELER_LOG_LOCAL(error)
                << "Failed to synchronize Stepper during destruction";
        }
    }
}

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
    CELER_VALIDATE(!step_in_flight_,
                   << "cannot warm up while a step is in flight");
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
 * Launch transport of already-initialized states.
 *
 * A single transport step is simply a loop over a topologically sorted DAG
 * of kernels. The step result must be retrieved with \c complete before
 * another step can be launched. In device mode the result counters are copied
 * asynchronously to pinned host memory, followed by a completion event.
 *
 * Existing synchronization within the action sequence can still block this
 * call. Removing those counter-dependent synchronization points is handled
 * separately.
 */
template<MemSpace M>
void Stepper<M>::launch()
{
    CELER_VALIDATE(!step_in_flight_,
                   << "cannot launch while a step is in flight");

    ScopedProfiling profile_this{"step"};
    auto counters = state_->sync_get_counters();
    counters.num_generated = 0;
    counters.num_cut = 0;
    counters.num_errored = 0;
    state_->sync_put_counters(counters);
    actions_->step(*params_, *state_);

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
    step_in_flight_ = true;
}

//---------------------------------------------------------------------------//
/*!
 * Whether the launched step has completed.
 */
template<MemSpace M>
bool Stepper<M>::ready() const
{
    CELER_VALIDATE(step_in_flight_,
                   << "cannot query completion without an in-flight step");
    return step_done_.ready();
}

//---------------------------------------------------------------------------//
/*!
 * Wait for and return the launched step result.
 *
 * Calling this consumes the pending result and allows another step to be
 * launched.
 */
template<MemSpace M>
auto Stepper<M>::complete() -> result_type
{
    CELER_VALIDATE(step_in_flight_,
                   << "cannot complete without an in-flight step");

    step_done_.sync();
    auto result = make_stepper_result(result_counters_.front());
    step_in_flight_ = false;
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Transport already-initialized states.
 *
 * This is the synchronous compatibility wrapper for \c launch and \c complete.
 */
template<MemSpace M>
auto Stepper<M>::operator()() -> result_type
{
    this->launch();
    return this->complete();
}

//---------------------------------------------------------------------------//
/*!
 * Initialize new primaries and transport them for a single step.
 */
template<MemSpace M>
auto Stepper<M>::operator()(SpanConstPrimary primaries) -> result_type
{
    CELER_VALIDATE(!step_in_flight_,
                   << "cannot transport primaries while a step is in flight");
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
    counters.num_pending = primaries.size();
    state_->sync_put_counters(counters);
    primaries_action_->insert(*params_, *state_, primaries);

    return (*this)();
}

//---------------------------------------------------------------------------//
/*!
 * Kill all tracks in flight to debug "stuck" tracks.
 *
 * The next "step" will apply the tracking cut and (if CPU) print diagnostic
 * output about the failed tracks.
 */
template<MemSpace M>
void Stepper<M>::kill_active()
{
    CELER_VALIDATE(!step_in_flight_,
                   << "cannot kill active tracks while a step is in flight");
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
    CELER_VALIDATE(!step_in_flight_,
                   << "cannot reseed while a step is in flight");
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
    CELER_VALIDATE(!step_in_flight_,
                   << "cannot reset state while a step is in flight");
    state_->reset();
}

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//

template class Stepper<MemSpace::host>;
template class Stepper<MemSpace::device>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
