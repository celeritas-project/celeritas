//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/detail/ActionSequence.cc
//---------------------------------------------------------------------------//
#include "ActionSequence.hh"

#include <algorithm>
#include <type_traits>
#include <utility>

#include "corecel/DeviceRuntimeApi.hh"

#include "corecel/Types.hh"
#include "corecel/cont/EnumArray.hh"
#include "corecel/cont/Range.hh"
#include "corecel/sys/ActionRegistry.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "corecel/sys/Stopwatch.hh"
#include "corecel/sys/Stream.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/track/StatusChecker.hh"

#include "../ActionInterface.hh"
#include "../CoreState.hh"
#include "../Debug.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct from an action registry and sequence options.
 */
template<class P, template<MemSpace M> class S>
ActionSequence<P, S>::ActionSequence(ActionRegistry const& reg, Options options)
    : options_(std::move(options))
{
    actions_.reserve(reg.num_actions());
    // Loop over all action IDs
    for (auto aidx : range(reg.num_actions()))
    {
        // Get abstract action shared pointer to determine type
        auto const& base = reg.action(ActionId{aidx});
        static_assert(std::is_same_v<StepActionT, CoreStepActionInterface>);
        if (auto step_act = std::dynamic_pointer_cast<StepActionT const>(base))
        {
            // Add stepping action to our array
            actions_.push_back(std::move(step_act));
        }
    }

    // Loop over all mutable actions
    for (auto const& base : reg.mutable_actions())
    {
        if (auto brun
            = std::dynamic_pointer_cast<CoreBeginRunActionInterface>(base))
        {
            // Add beginning-of-run to the array
            begin_run_.emplace_back(std::move(brun));
        }
        if (auto erun
            = std::dynamic_pointer_cast<CoreEndRunGatherActionInterface>(base))
        {
            // Add end-of-run to the array
            end_run_.emplace_back(std::move(erun));
        }
    }

    // Sort actions by increasing order (and secondarily, increasing IDs)
    std::sort(actions_.begin(),
              actions_.end(),
              [](SPConstStepAction const& a, SPConstStepAction const& b) {
                  return OrderedAction{a->order(), a->action_id()}
                         < OrderedAction{b->order(), b->action_id()};
              });

    // Initialize timing
    accum_time_.resize(actions_.size());

    // Get status checker if available
    for (auto const& brun_sp : begin_run_)
    {
        if (auto sc = std::dynamic_pointer_cast<StatusChecker>(brun_sp))
        {
            // Add status checker
            status_checker_ = std::move(sc);
            CELER_LOG(info) << "Executing actions with additional debug "
                               "checking";
            break;
        }
    }

    CELER_ENSURE(actions_.size() == accum_time_.size());
}

//---------------------------------------------------------------------------//
/*!
 * Initialize actions and states.
 */
template<class P, template<MemSpace M> class S>
template<MemSpace M>
void ActionSequence<P, S>::begin_run(P const& params, S<M>& state)
{
    for (auto const& sp_action : begin_run_)
    {
        ScopedProfiling profile_this{sp_action->label()};
        sp_action->begin_run(params, state);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Merge states at the end of the run
 */
template<class P, template<MemSpace M> class S>
template<MemSpace M>
void ActionSequence<P, S>::end_run_gather(P const& params,
                                          Span<S<M>* const> states)
{
    CELER_EXPECT(std::all_of(states.begin(), states.end(), [](auto* s) {
        return static_cast<bool>(s);
    }));
    for (auto const& sp_action : end_run_)
    {
        ScopedProfiling profile_this{sp_action->label()};
        sp_action->end_run_gather(params, states);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Call all explicit actions with host or device data.
 */
template<class P, template<MemSpace M> class S>
template<MemSpace M>
void ActionSequence<P, S>::step(P const& params, S<M>& state)
{
    [[maybe_unused]] Stream::StreamT stream = nullptr;
    if (M == MemSpace::device && options_.action_times)
    {
        stream = celeritas::device().stream(state.stream_id()).get();
    }

    if constexpr (M == MemSpace::host && std::is_same_v<CoreParams, P>)
    {
        if (status_checker_)
        {
            g_debug_executing_params = &params;
        }
    }

    // Running a single track slot on host:
    // Skip inapplicable post-step action
    auto const skip_post_action = [&](auto const& action) {
        if constexpr (M != MemSpace::host)
        {
            return false;
        }
        return state.size() == 1 && action.order() == StepActionOrder::post
               && action.action_id()
                      != state.ref().sim.post_step_action[TrackSlotId{0}];
    };

    if (options_.action_times && !state.warming_up())
    {
        // Execute all actions and record the time elapsed
        for (auto i : range(actions_.size()))
        {
            if (auto const& action = *actions_[i]; !skip_post_action(action))
            {
                ScopedProfiling profile_this{action.label()};
                Stopwatch get_time;
                action.step(params, state);
                if constexpr (M == MemSpace::device)
                {
                    CELER_DEVICE_CALL_PREFIX(StreamSynchronize(stream));
                }
                accum_time_[i] += get_time();
                if (CELER_UNLIKELY(status_checker_))
                {
                    status_checker_->step(action.action_id(), params, state);
                }
            }
        }
    }
    else
    {
        // Just loop over the actions
        for (auto const& sp_action : actions_)
        {
            if (auto const& action = *sp_action; !skip_post_action(action))
            {
                ScopedProfiling profile_this{action.label()};
                action.step(params, state);
                if (CELER_UNLIKELY(status_checker_))
                {
                    status_checker_->step(action.action_id(), params, state);
                }
            }
        }
    }

    if (M == MemSpace::host && std::is_same_v<CoreParams, P> && status_checker_)
    {
        g_debug_executing_params = nullptr;
    }
}

//---------------------------------------------------------------------------//
// Explicit template instantiation
//---------------------------------------------------------------------------//

template class ActionSequence<CoreParams, CoreState>;

template void
ActionSequence<CoreParams, CoreState>::begin_run(CoreParams const&,
                                                 CoreState<MemSpace::host>&);
template void
ActionSequence<CoreParams, CoreState>::begin_run(CoreParams const&,
                                                 CoreState<MemSpace::device>&);

template void
ActionSequence<CoreParams, CoreState>::step(CoreParams const&,
                                            CoreState<MemSpace::host>&);
template void
ActionSequence<CoreParams, CoreState>::step(CoreParams const&,
                                            CoreState<MemSpace::device>&);

// TODO: add explicit template instantiation of execute for optical data

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
