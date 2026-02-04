//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detail/OpticalLaunchAction.cc
//---------------------------------------------------------------------------//
#include "OpticalLaunchAction.hh"

#include <mutex>

#include "corecel/data/AuxParamsRegistry.hh"
#include "corecel/data/AuxStateVec.hh"
#include "corecel/io/Logger.hh"
#include "corecel/sys/ActionRegistry.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/phys/GeneratorRegistry.hh"

#include "../CoreParams.hh"
#include "../CoreState.hh"
#include "../Transporter.hh"
#include "../action/ActionGroups.hh"
#include "../gen/GeneratorData.hh"

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
    : action_id_{action_id}, aux_id_{aux_id}, data_(std::move(input))
{
    CELER_EXPECT(data_);
}

//---------------------------------------------------------------------------//
/*!
 * Create the action groups and get a pointer to the aux data.
 */
void OpticalLaunchAction::begin_run(CoreParams const&, CoreStateHost& state)
{
    this->begin_run_impl(state);
}

//---------------------------------------------------------------------------//
/*!
 * Create the action groups and get a pointer to the aux data.
 */
void OpticalLaunchAction::begin_run(CoreParams const&, CoreStateDevice& state)
{
    this->begin_run_impl(state);
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
            this->optical_params(), sid, this->state_size());
    }
    else if (m == MemSpace::device)
    {
        return std::make_unique<optical::CoreState<MemSpace::device>>(
            this->optical_params(), sid, this->state_size());
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

    auto const core_counters = core_state.sync_get_counters();
    auto const counters = state.sync_get_counters();

    if ((counters.num_pending < data_.auto_flush
         && (core_counters.num_alive > 0 || core_counters.num_initializers > 0))
        || counters.num_pending == 0)
    {
        // Don't launch the optical loop if the number of pending tracks is
        // below the threshold and the core stepping loop hasn't completed yet
        return;
    }

    // Transport pending optical tracks
    (*transport_)(state);
}

//---------------------------------------------------------------------------//
/*!
 * Create the transporter and cache a pointer to the auxiliary data.
 *
 * This allows additional optical actions to be added after the launch action
 * has been constructed.
 */
template<MemSpace M>
void OpticalLaunchAction::begin_run_impl(CoreState<M>& core_state)
{
    if (!transport_)
    {
        static std::mutex launch_mutex;
        std::lock_guard<std::mutex> scoped_lock{launch_mutex};

        if (!transport_)
        {
            // Create the transporter
            optical::Transporter::Input inp;
            inp.params = data_.optical_params;
            inp.action_times = data_.action_times;
            transport_ = std::make_shared<optical::Transporter>(std::move(inp));
        }
    }

    // Store a pointer to the auxiliary state vector
    auto& state = get<optical::CoreState<M>>(core_state.aux(), this->aux_id());
    state.aux() = core_state.aux_ptr();

    CELER_ENSURE(transport_);
    CELER_ENSURE(state.aux());
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
