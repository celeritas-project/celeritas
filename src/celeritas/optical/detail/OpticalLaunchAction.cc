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
#include "corecel/sys/ActionRegistry.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/TrackInitParams.hh"
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
                                     SPOffloadParams offload)
{
    CELER_EXPECT(material);
    CELER_EXPECT(offload);
    ActionRegistry& actions = *core.action_reg();
    AuxParamsRegistry& aux = *core.aux_reg();
    auto result = std::make_shared<OpticalLaunchAction>(actions.next_id(),
                                                        aux.next_id(),
                                                        core,
                                                        std::move(material),
                                                        std::move(offload));

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
                                         SPOffloadParams offload)
    : action_id_{action_id}
    , aux_id_{data_id}
    , offload_params_{std::move(offload)}
{
    CELER_EXPECT(material);
    CELER_EXPECT(offload_params_);

    // Create optical core params
    optical_params_ = std::make_shared<optical::CoreParams>([&] {
        optical::CoreParams::Input inp;
        inp.geometry = core.geometry();
        inp.material = std::move(material);
        // TODO: unique RNG streams for optical loop
        inp.rng = core.rng();
        inp.sim = std::make_shared<SimParams>();
        // TODO: get capacity from input
        inp.init = std::make_shared<optical::TrackInitParams>(
            core.init()->host_ref().capacity);
        inp.action_reg = std::make_shared<ActionRegistry>();
        inp.max_streams = core.max_streams();
        CELER_ENSURE(inp);
        return inp;
    }());
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
void OpticalLaunchAction::execute_impl(CoreParams const& core_params,
                                       CoreState<M>& core_state) const
{
    auto& offload_state = get<OpticalOffloadState<M>>(
        core_state.aux(), offload_params_->aux_id());
    auto& optical_state
        = get<optical::CoreState<M>>(core_state.aux(), this->aux_id());

    // Loop!
    CELER_ASSERT(offload_state);
    CELER_ASSERT(optical_state.size() > 0);
    CELER_DISCARD(core_params);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
