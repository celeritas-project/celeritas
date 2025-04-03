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
#include "corecel/io/OutputInterfaceAdapter.hh"
#include "corecel/io/OutputRegistry.hh"
#include "corecel/sys/ActionRegistry.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/track/TrackInitParams.hh"

#include "OffloadParams.hh"
#include "OpticalSizes.json.hh"
#include "../CoreParams.hh"
#include "../CoreState.hh"
#include "../PhysicsParams.hh"
#include "../TrackInitParams.hh"
#include "../action/ActionGroups.hh"

namespace celeritas
{
namespace detail
{
namespace
{
//---------------------------------------------------------------------------//
auto get_core_sizes(OpticalLaunchAction const& ola)
{
    // Optical core params
    auto const& cp = ola.optical_params();

    OpticalSizes result;
    result.streams = cp.max_streams();

    // NOTE: quantities are *per-process* quantities: integrated over streams,
    // but not processes
    result.generators = result.streams
                        * ola.offload_params().host_ref().setup.capacity;
    result.initializers = result.streams * cp.init()->capacity();
    result.tracks = result.streams * ola.state_size();

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace

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
        actions.next_id(), aux.next_id(), core, std::move(input));

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
                                         Input&& input)
    : action_id_{action_id}
    , aux_id_{data_id}
    , offload_params_{std::move(input.offload)}
    , state_size_{input.num_track_slots}
{
    CELER_EXPECT(offload_params_);
    CELER_EXPECT(state_size_ > 0);
    CELER_EXPECT(input.material);
    CELER_EXPECT(input.initializer_capacity > 0);

    // Create optical core params
    optical_params_ = std::make_shared<optical::CoreParams>([&] {
        optical::CoreParams::Input inp;
        inp.geometry = core.geometry();
        inp.material = std::move(input.material);
        // TODO: unique RNG streams for optical loop
        inp.rng = core.rng();
        inp.init = std::make_shared<optical::TrackInitParams>(
            input.initializer_capacity);
        inp.action_reg = std::make_shared<ActionRegistry>();
        inp.max_streams = core.max_streams();
        {
            optical::PhysicsParams::Input phys_input;
            phys_input.model_builders = std::move(input.model_builders);
            phys_input.materials = inp.material;
            phys_input.action_registry = inp.action_reg.get();
            inp.physics = std::make_shared<optical::PhysicsParams>(
                std::move(phys_input));
        }
        CELER_ENSURE(inp);
        return inp;
    }());

    // Add optical sizes
    core.output_reg()->insert(
        OutputInterfaceAdapter<detail::OpticalSizes>::from_rvalue_ref(
            OutputInterface::Category::internal,
            "optical-sizes",
            get_core_sizes(*this)));

    // TODO: add generators to the *optical* stepping loop instead of part of
    // the main loop; for now just make sure enough track initializers are
    // allocated so that we can initialize them all at the beginning of step

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
auto OpticalLaunchAction::create_state(MemSpace m,
                                       StreamId sid,
                                       size_type) const -> UPState
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
    auto& offload_state = get<OpticalOffloadState<M>>(
        core_state.aux(), offload_params_->aux_id());
    auto& optical_state
        = get<optical::CoreState<M>>(core_state.aux(), this->aux_id());
    CELER_ASSERT(offload_state);
    CELER_ASSERT(optical_state.size() > 0);

    constexpr size_type max_step_iters{1024};
    size_type num_step_iters{0};
    size_type num_steps{0};

    // Loop while photons are yet to be tracked
    auto& counters = optical_state.counters();
    auto const& step_actions = optical_actions_->step();
    while (counters.num_initializers > 0 || counters.num_alive > 0)
    {
        // Loop through actions
        for (auto const& action : step_actions)
        {
            action->step(*optical_params_, optical_state);
        }

        num_steps += counters.num_active;
        if (CELER_UNLIKELY(++num_step_iters == max_step_iters))
        {
            CELER_LOG_LOCAL(error)
                << "Exceeded step count of " << max_step_iters
                << ": aborting optical transport loop with "
                << counters.num_active << " active tracks, "
                << counters.num_alive << " alive tracks, "
                << counters.num_vacancies << " vacancies, and "
                << counters.num_initializers << " queued";
            break;
        }
    }

    // Update statistics
    offload_state.accum.steps += num_steps;
    offload_state.accum.step_iters += num_step_iters;
    ++offload_state.accum.flushes;

    // TODO: generation is done *outside* of the optical tracking loop;
    // once we move it inside, update the generation count in the loop here
    // TODO: is this correct if we abort the tracking loop early?
    counters.num_generated = 0;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
