//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/Stepper.cc
//---------------------------------------------------------------------------//
#include "Stepper.hh"

#include <utility>

#include "corecel/cont/Range.hh"
#include "corecel/data/Ref.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "orange/OrangeData.hh"
#include "celeritas/Types.hh"
#include "celeritas/random/RngParams.hh"
#include "celeritas/random/RngReseed.hh"
#include "celeritas/track/TrackInitParams.hh"

#include "CoreParams.hh"
#include "detail/ActionSequence.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with problem parameters and setup options.
 */
template<MemSpace M>
Stepper<M>::Stepper(Input input)
    : params_(std::move(input.params))
    , state_(*params_, input.stream_id, input.num_track_slots)
{
    // Create action sequence
    actions_ = [&] {
        ActionSequence::Options opts;
        opts.sync = input.sync;
        return std::make_shared<ActionSequence>(*params_->action_reg(), opts);
    }();

    // Execute beginning-of-run action
    ScopedProfiling profile_this{"begin-run"};
    actions_->begin_run(*params_, state_);
}

//---------------------------------------------------------------------------//
//! Default destructor
template<MemSpace M>
Stepper<M>::~Stepper() = default;

//---------------------------------------------------------------------------//
/*!
 * Transport already-initialized states.
 *
 * A single transport step is simply a loop over a toplogically sorted DAG
 * of kernels.
 */
template<MemSpace M>
auto Stepper<M>::operator()() -> result_type
{
    ScopedProfiling profile_this{"step"};
    actions_->execute(*params_, state_);

    // Get the number of track initializers and active tracks
    result_type result;
    result.active = state_.counters().num_active;
    result.alive = state_.counters().num_alive;
    result.queued = state_.counters().num_initializers;

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Initialize new primaries and transport them for a single step.
 */
template<MemSpace M>
auto Stepper<M>::operator()(SpanConstPrimary primaries) -> result_type
{
    CELER_EXPECT(!primaries.empty());

    // Check initializer capacity
    size_type num_initializers = state_.counters().num_initializers;
    size_type init_capacity = this->state_ref().init.initializers.size();
    CELER_VALIDATE(primaries.size() + num_initializers <= init_capacity,
                   << "insufficient initializer capacity (" << init_capacity
                   << ") with size (" << num_initializers
                   << ") for primaries (" << primaries.size() << ")");

    // Check that events are consistent with our 'max events'
    auto max_id
        = std::max_element(primaries.begin(),
                           primaries.end(),
                           [](Primary const& left, Primary const& right) {
                               return left.event_id < right.event_id;
                           });
    CELER_VALIDATE(max_id->event_id < params_->init()->max_events(),
                   << "event number " << max_id->event_id.unchecked_get()
                   << " exceeds max_events=" << params_->init()->max_events());

    CELER_ASSERT(state_.primary_range().empty());
    state_.insert_primaries(primaries);

    return (*this)();
}

//---------------------------------------------------------------------------//
/*!
 * Reseed the RNGs at the start of an event for "strong" reproducibility.
 *
 * This reinitializes the RNG states using a single seed and unique subsequence
 * for each thread. It ensures that given an event number, that event can be
 * reproduced.
 */
template<MemSpace M>
void Stepper<M>::reseed(EventId event_id)
{
    reseed_rng(get_ref<M>(*params_->rng()), state_.ref().rng, event_id.get());
}

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//

template class Stepper<MemSpace::host>;
template class Stepper<MemSpace::device>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
