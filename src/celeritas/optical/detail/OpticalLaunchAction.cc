//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detail/OpticalLaunchAction.cc
//---------------------------------------------------------------------------//
#include "OpticalLaunchAction.hh"

#include "corecel/data/AuxParamsRegistry.hh"
#include "corecel/data/AuxStateVec.hh"
#include "corecel/io/Logger.hh"
#include "corecel/sys/ActionRegistry.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"

#include "../CoreParams.hh"
#include "../CoreState.hh"
#include "../action/ActionGroups.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct and add to core params.
 */
std::shared_ptr<OpticalLaunchAction>
OpticalLaunchAction::make_and_insert(CoreParams const& core, Input&& input)
{
    CELER_EXPECT(input);
    ActionRegistry& actions = *core.action_reg();
    AuxParamsRegistry& aux = *core.aux_reg();
    auto result = std::make_shared<OpticalLaunchAction>(
        actions.next_id(), aux.next_id(), std::move(input));

    actions.insert(result);
    aux.insert(result);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Construct with action ID, generator storage.
 */
OpticalLaunchAction::OpticalLaunchAction(ActionId action_id,
                                         AuxId aux_id,
                                         Input&& input)
    : action_id_{action_id}
    , aux_id_{aux_id}
    , optical_params_(std::move(input.optical_params))
    , state_size_(input.num_track_slots)
    , max_step_iters_(input.max_step_iters)
    , auto_flush_(input.auto_flush)
{
    CELER_EXPECT(optical_params_);
    CELER_EXPECT(state_size_ > 0);

    // TODO: Generate optical photons directly in the track slots rather than a
    // buffer. For now just make sure enough track initializers are allocated
    // so that we can initialize them all at the start of the first step

    // TODO: should we initialize this at begin-run so that we can add
    // additional optical actions?
    optical_actions_
        = std::make_shared<ActionGroupsT>(*optical_params_->action_reg());
}

//---------------------------------------------------------------------------//
/*!
 * Descriptive name of the action.
 */
std::string_view OpticalLaunchAction::description() const
{
    return "launch the optical stepping loop";
}

//---------------------------------------------------------------------------//
/*!
 * Build state data for a stream.
 */
auto OpticalLaunchAction::create_state(MemSpace m, StreamId sid, size_type) const
    -> UPState
{
    if (m == MemSpace::host)
    {
        return std::make_unique<optical::CoreState<MemSpace::host>>(
            *optical_params_, sid, state_size_);
    }
    else if (m == MemSpace::device)
    {
        return std::make_unique<optical::CoreState<MemSpace::device>>(
            *optical_params_, sid, state_size_);
    }
    CELER_ASSERT_UNREACHABLE();
}

//---------------------------------------------------------------------------//
/*!
 * Perform a step action with host data.
 */
void OpticalLaunchAction::step(CoreParams const& params,
                               CoreStateHost& state) const
{
    return this->execute_impl(params, state);
}

//---------------------------------------------------------------------------//
/*!
 * Perform a step action with device data.
 */
void OpticalLaunchAction::step(CoreParams const& params,
                               CoreStateDevice& state) const
{
    return this->execute_impl(params, state);
}

//---------------------------------------------------------------------------//
/*!
 * Launch the optical tracking loop.
 */
template<MemSpace M>
void OpticalLaunchAction::execute_impl(CoreParams const&,
                                       CoreState<M>& core_state) const
{
    auto& state = get<optical::CoreState<M>>(core_state.aux(), this->aux_id());
    CELER_ASSERT(state.size() > 0);

    size_type num_step_iters{0};
    size_type num_steps{0};

    if (!state.aux())
    {
        // Get a pointer to the auxiliary state vector
        state.aux() = core_state.aux_ptr();
    }

    auto const& core_counters = core_state.counters();
    auto& counters = state.counters();
    CELER_ASSERT(counters.num_initializers == 0);

    if ((counters.num_pending < auto_flush_
         && (core_counters.num_alive > 0 || core_counters.num_initializers > 0))
        || max_step_iters_ == 0)
    {
        // Don't launch the optical loop if the number of pending tracks is
        // below the threshold and the core stepping loop hasn't completed yet
        return;
    }

    auto init_capacity = state.ref().init.initializers.size();
    CELER_VALIDATE(counters.num_pending <= init_capacity,
                   << "insufficient capacity (" << init_capacity
                   << ") for optical photon initializers (total capacity "
                      "requirement of "
                   << counters.num_pending << ")");

    // Loop while photons are yet to be tracked
    auto const& step_actions = optical_actions_->step();
    while (counters.num_pending > 0 || counters.num_initializers > 0
           || counters.num_alive > 0)
    {
        // Loop through actions
        for (auto const& action : step_actions)
        {
            action->step(*optical_params_, state);
        }

        num_steps += counters.num_active;
        if (CELER_UNLIKELY(++num_step_iters == max_step_iters_))
        {
            CELER_LOG_LOCAL(error)
                << "Exceeded step count of " << max_step_iters_
                << ": aborting optical transport loop with "
                << counters.num_generated << " generated tracks, "
                << counters.num_active << " active tracks, "
                << counters.num_alive << " alive tracks, "
                << counters.num_vacancies << " vacancies, and "
                << counters.num_initializers << " queued";

            state.reset();
            break;
        }
    }

    // Update statistics
    state.accum().steps += num_steps;
    state.accum().step_iters += num_step_iters;
    ++state.accum().flushes;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
