//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/ActionSequence.cc
//---------------------------------------------------------------------------//
#include "ActionSequence.hh"

#include <algorithm>
#include <type_traits>
#include <utility>
#include <vector>

#include "corecel/DeviceRuntimeApi.hh"

#include "corecel/Types.hh"
#include "corecel/cont/EnumArray.hh"
#include "corecel/cont/Range.hh"
#include "corecel/sys/ActionRegistry.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "corecel/sys/Stopwatch.hh"
#include "corecel/sys/Stream.hh"
#include "celeritas/track/StatusChecker.hh"

#include "ActionInterface.hh"
#include "CoreParams.hh"
#include "CoreState.hh"
#include "Debug.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from an action registry and sequence options.
 */
ActionSequence::ActionSequence(ActionRegistry const& reg, Options options)
    : actions_{reg}
    , options_{std::move(options)}
    , num_actions_(reg.num_actions())
{
    // Get status checker if available
    for (auto const& brun_sp : actions_.begin_run())
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
}

//---------------------------------------------------------------------------//
/*!
 * Initialize actions and states.
 */
template<MemSpace M>
void ActionSequence::begin_run(CoreParams const& params, CoreState<M>& state)
{
    CELER_VALIDATE(params.action_reg()->num_actions() == num_actions_,
                   << "Number of actions changed since setup completed");

    for (auto const& sp_action : actions_.begin_run())
    {
        ScopedProfiling profile_this{sp_action->label()};
        sp_action->begin_run(params, state);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Call all explicit actions with host or device data.
 */
template<MemSpace M>
void ActionSequence::step(CoreParams const& params, CoreState<M>& state)
{
    // Save a pointer to aux data and the stream for synchronizing after each
    // step for timing
    // NOTE: instead of synchronizing the stream we could add device timers to
    // reduce the performance impact
    Stream* stream = nullptr;
    std::vector<double>* accum_time = nullptr;
    if (options_.action_times)
    {
        accum_time = &options_.action_times->state(state.aux()).accum_time;
        if (M == MemSpace::device)
        {
            stream = &device().stream(state.stream_id());
        }
    }

    // When running a single track slot on host, we can preemptively skip
    // inapplicable post-step actions
    auto const skip_post_action = [&](auto const& action) {
        if constexpr (M != MemSpace::host)
        {
            return false;
        }
        return state.size() == 1 && action.order() == StepActionOrder::post
               && action.action_id()
                      != state.ref().sim.post_step_action[TrackSlotId{0}];
    };

    auto step_actions = make_span(actions_.step());
    if (options_.action_times && !state.warming_up())
    {
        // Execute all actions and record the time elapsed
        for (auto i : range(step_actions.size()))
        {
            if (auto const& action = *step_actions[i];
                !skip_post_action(action))
            {
                ScopedProfiling profile_this{action.label()};
                Stopwatch get_time;
                action.step(params, state);
                if (stream)
                {
                    stream->sync();
                }
                (*accum_time)[action.action_id().get()] += get_time();
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
        for (auto const& sp_action : actions_.step())
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
}

//---------------------------------------------------------------------------//
/*!
 * Get the accumulated action times.
 */
auto ActionSequence::get_action_times(AuxStateVec const& aux) const -> MapStrDbl
{
    if (options_.action_times)
    {
        return options_.action_times->get_action_times(aux);
    }
    return {};
}

//---------------------------------------------------------------------------//

template void
ActionSequence::begin_run(CoreParams const&, CoreState<MemSpace::host>&);
template void
ActionSequence::begin_run(CoreParams const&, CoreState<MemSpace::device>&);

template void
ActionSequence::step(CoreParams const&, CoreState<MemSpace::host>&);
template void
ActionSequence::step(CoreParams const&, CoreState<MemSpace::device>&);

//---------------------------------------------------------------------------//
}  // namespace celeritas
