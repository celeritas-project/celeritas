//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/detail/ActionSequence.cc
//---------------------------------------------------------------------------//
#include "ActionSequence.hh"

#include <algorithm>
#include <tuple>
#include <type_traits>
#include <utility>

#include "corecel/device_runtime_api.h"
#include "corecel/Types.hh"
#include "corecel/cont/EnumArray.hh"
#include "corecel/cont/Range.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "corecel/sys/Stopwatch.hh"
#include "corecel/sys/Stream.hh"

#include "ParamsTraits.hh"
#include "../ActionInterface.hh"
#include "../ActionRegistry.hh"
#include "../CoreState.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct from an action registry and sequence options.
 */
ActionSequence::ActionSequence(ActionRegistry const& reg, Options options)
    : options_(std::move(options))
{
    // Loop over all action IDs
    for (auto aidx : range(reg.num_actions()))
    {
        // Get abstract action shared pointer and see if it's explicit
        auto const& base = reg.action(ActionId{aidx});
        if (auto expl
            = std::dynamic_pointer_cast<ExplicitActionInterface const>(base))
        {
            // Add explicit action to our array
            actions_.push_back(std::move(expl));
        }
    }

    // Loop over all mutable actions
    for (auto const& base : reg.mutable_actions())
    {
        if (auto brun = std::dynamic_pointer_cast<BeginRunActionInterface>(base))
        {
            // Add beginning-of-run to the array
            begin_run_.push_back(std::move(brun));
        }
    }

    // Sort actions by increasing order (and secondarily, increasing IDs)
    std::sort(actions_.begin(),
              actions_.end(),
              [](SPConstExplicit const& a, SPConstExplicit const& b) {
                  return std::make_tuple(a->order(), a->action_id())
                         < std::make_tuple(b->order(), b->action_id());
              });

    // Initialize timing
    accum_time_.resize(actions_.size());

    CELER_ENSURE(actions_.size() == accum_time_.size());
}

//---------------------------------------------------------------------------//
/*!
 * Initialize actions and states.
 */
template<MemSpace M>
void ActionSequence::begin_run(CoreParams const& params, CoreState<M>& state)
{
    for (auto const& sp_action : begin_run_)
    {
        ScopedProfiling profile_this{sp_action->label()};
        sp_action->begin_run(params, state);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Call all explicit actions with host or device data.
 */
template<typename Params, template<MemSpace M> class State, MemSpace M>
void ActionSequence::execute(Params const& params, State<M>& state)
{
    using ExplicitAction = typename ParamsTraits<Params>::ExplicitAction;

    static_assert(
        std::is_same_v<State<M>, typename ParamsTraits<Params>::template State<M>>,
        "The Params and State type are not matching.");

    [[maybe_unused]] Stream::StreamT stream = nullptr;
    if (M == MemSpace::device && options_.sync)
    {
        stream = celeritas::device().stream(state.stream_id()).get();
    }

    if ((M == MemSpace::host || options_.sync) && !state.warming_up())
    {
        // Execute all actions and record the time elapsed
        for (auto i : range(actions_.size()))
        {
            ScopedProfiling profile_this{actions_[i]->label()};
            Stopwatch get_time;
            auto const& concrete_action
                = dynamic_cast<ExplicitAction const&>(*actions_[i]);
            concrete_action.execute(params, state);
            if (M == MemSpace::device)
            {
                CELER_DEVICE_CALL_PREFIX(StreamSynchronize(stream));
            }
            accum_time_[i] += get_time();
        }
    }
    else
    {
        // Just loop over the actions
        for (SPConstExplicit const& sp_action : actions_)
        {
            ScopedProfiling profile_this{sp_action->label()};
            auto const& concrete_action
                = dynamic_cast<ExplicitAction const&>(*sp_action);
            concrete_action.execute(params, state);
        }
    }
}

//---------------------------------------------------------------------------//
// Explicit template instantiation
//---------------------------------------------------------------------------//

template void
ActionSequence::begin_run(CoreParams const&, CoreState<MemSpace::host>&);
template void
ActionSequence::begin_run(CoreParams const&, CoreState<MemSpace::device>&);

template void
ActionSequence::execute(CoreParams const&, CoreState<MemSpace::host>&);
template void
ActionSequence::execute(CoreParams const&, CoreState<MemSpace::device>&);

// TODO: add explicit template instantiation of execute for optical data

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
