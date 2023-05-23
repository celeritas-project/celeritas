//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/Stepper.cc
//---------------------------------------------------------------------------//
#include "Stepper.hh"

#include <type_traits>
#include <utility>

#include "corecel/cont/Range.hh"
#include "corecel/data/Ref.hh"
#include "orange/OrangeData.hh"
#include "celeritas/Types.hh"
#include "celeritas/random/XorwowRngData.hh"
#include "celeritas/track/TrackInitData.hh"
#include "celeritas/track/TrackInitUtils.hh"
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
    actions_->execute(*params_, state_);

    // Get the number of track initializers and active tracks
    auto const& init = this->state_ref().init;
    result_type result;
    result.active = init.scalars.num_active;
    result.alive = init.scalars.num_alive;
    result.queued = init.scalars.num_initializers;

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

    size_type num_initializers
        = this->state_ref().init.scalars.num_initializers;
    size_type init_capacity = this->state_ref().init.initializers.size();
    CELER_VALIDATE(primaries.size() + num_initializers <= init_capacity,
                   << "insufficient initializer capacity (" << init_capacity
                   << ") with size (" << num_initializers
                   << ") for primaries (" << primaries.size() << ")");
    auto max_id
        = std::max_element(primaries.begin(),
                           primaries.end(),
                           [](Primary const& left, Primary const& right) {
                               return left.event_id < right.event_id;
                           });
    CELER_VALIDATE(max_id->event_id < params_->init()->max_events(),
                   << "event number " << max_id->event_id.unchecked_get()
                   << " exceeds max_events=" << params_->init()->max_events());

    // Create track initializers
    extend_from_primaries(*params_, state_, primaries);

    return (*this)();
}

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//

template class Stepper<MemSpace::host>;
template class Stepper<MemSpace::device>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
