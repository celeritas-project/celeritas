//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
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
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/TrackInitParams.hh"
#include "celeritas/optical/action/ActionGroups.hh"
#include "celeritas/track/SimParams.hh"
#include "celeritas/track/TrackInitParams.hh"

#include "OffloadParams.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct and add to core params.
 */
std::shared_ptr<OpticalLaunchAction>
OpticalLaunchAction::make_and_insert(CoreParams const& core,
                                     SPConstMaterial material,
                                     SPOffloadParams offload,
                                     size_type primary_capacity)
{
    CELER_EXPECT(material);
    CELER_EXPECT(offload);
    ActionRegistry& actions = *core.action_reg();
    AuxParamsRegistry& aux = *core.aux_reg();
    auto result = std::make_shared<OpticalLaunchAction>(actions.next_id(),
                                                        aux.next_id(),
                                                        core,
                                                        std::move(material),
                                                        std::move(offload),
                                                        primary_capacity);

    actions.insert(result);
    aux.insert(result);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Construct with action ID, generator storage.
 */
OpticalLaunchAction::OpticalLaunchAction(ActionId action_id,
                                         AuxId data_id,
                                         CoreParams const& core,
                                         SPConstMaterial material,
                                         SPOffloadParams offload,
                                         size_type primary_capacity)
    : action_id_{action_id}
    , aux_id_{data_id}
    , offload_params_{std::move(offload)}
{
    CELER_EXPECT(material);
    CELER_EXPECT(offload_params_);
    CELER_EXPECT(primary_capacity > 0);

    auto optical_actions = std::make_shared<ActionRegistry>();

    // Create optical core params
    optical_params_ = std::make_shared<optical::CoreParams>([&] {
        optical::CoreParams::Input inp;
        inp.geometry = core.geometry();
        inp.material = std::move(material);
        // TODO: unique RNG streams for optical loop
        inp.rng = core.rng();
        inp.sim = std::make_shared<SimParams>();
        inp.init = std::make_shared<optical::TrackInitParams>(primary_capacity);
        inp.action_reg = optical_actions;
        inp.max_streams = core.max_streams();
        CELER_ENSURE(inp);
        return inp;
    }());

    // TODO: move generators to the *optical* stepping loop; for now just make
    // sure enough track initializers are allocated

    // Add along-step

    // Add boundary action

    // TODO: should we initialize this at begin-run so that we can add
    // additional optical actions?
    optical_actions_ = std::make_shared<ActionGroupsT>(*optical_actions);
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
auto OpticalLaunchAction::create_state(MemSpace m,
                                       StreamId sid,
                                       size_type size) const -> UPState
{
    if (m == MemSpace::host)
    {
        return std::make_unique<optical::CoreState<MemSpace::host>>(
            *optical_params_, sid, size);
    }
    else if (m == MemSpace::device)
    {
        return std::make_unique<optical::CoreState<MemSpace::device>>(
            *optical_params_, sid, size);
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
    auto& offload_state = get<OpticalOffloadState<M>>(
        core_state.aux(), offload_params_->aux_id());
    auto& optical_state
        = get<optical::CoreState<M>>(core_state.aux(), this->aux_id());
    CELER_ASSERT(offload_state);
    CELER_ASSERT(optical_state.size() > 0);

    constexpr size_type max_steps{8};
    size_type remaining_steps = max_steps;

    // Loop while photons are yet to be tracked
    auto& counters = optical_state.counters();
    auto const& step_actions = optical_actions_->step();
    while (counters.num_initializers > 0 || counters.num_alive > 0)
    {
        // TODO: generation is done *outside* of the optical tracking loop;
        // once we move it inside, update the generation count in the
        // generators
        counters.num_generated = 0;

        // Loop through actions
        for (auto const& action : step_actions)
        {
            action->step(*optical_params_, optical_state);
        }
        CELER_LOG(debug) << "Stepped " << counters.num_active
                         << " optical tracks";

        if (CELER_UNLIKELY(--remaining_steps == 0))
        {
            CELER_LOG_LOCAL(error) << "Exceeded step count of " << max_steps
                                   << ": aborting optical transport loop with "
                                   << counters.num_alive << " tracks and "
                                   << counters.num_initializers << " queued";
            break;
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
