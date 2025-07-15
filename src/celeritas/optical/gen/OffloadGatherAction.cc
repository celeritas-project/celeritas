//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/OffloadGatherAction.cc
//---------------------------------------------------------------------------//
#include "OffloadGatherAction.hh"

#include <algorithm>

#include "corecel/Assert.hh"
#include "corecel/data/AuxParamsRegistry.hh"
#include "corecel/sys/ActionRegistry.hh"
#include "celeritas/global/ActionLauncher.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/CoreTrackData.hh"
#include "celeritas/global/TrackExecutor.hh"

#include "detail/OffloadGatherExecutor.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct and add to core params.
 */
std::shared_ptr<OffloadGatherAction>
OffloadGatherAction::make_and_insert(CoreParams const& core)
{
    ActionRegistry& actions = *core.action_reg();
    AuxParamsRegistry& aux = *core.aux_reg();
    auto result = std::make_shared<OffloadGatherAction>(actions.next_id(),
                                                        aux.next_id());
    actions.insert(result);
    aux.insert(result);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Construct with action ID and aux ID.
 */
OffloadGatherAction::OffloadGatherAction(ActionId action_id, AuxId aux_id)
    : action_id_(action_id), aux_id_(aux_id)
{
    CELER_EXPECT(action_id_);
    CELER_EXPECT(aux_id_);
}

//---------------------------------------------------------------------------//
/*!
 * Build state data for a stream.
 */
auto OffloadGatherAction::create_state(MemSpace m,
                                       StreamId id,
                                       size_type size) const -> UPState
{
    return make_aux_state<OffloadStepStateData>(m, id, size);
}

//---------------------------------------------------------------------------//
/*!
 * Descriptive name of the action.
 */
std::string_view OffloadGatherAction::description() const
{
    return "gather pre-step data to generate optical distributions";
}

//---------------------------------------------------------------------------//
/*!
 * Gather pre-step data.
 */
void OffloadGatherAction::step(CoreParams const& params,
                               CoreStateHost& state) const
{
    auto& step = state.aux_data<OffloadStepStateData>(aux_id_);
    auto execute
        = make_active_track_executor(params.ptr<MemSpace::native>(),
                                     state.ptr(),
                                     detail::OffloadGatherExecutor{step});
    launch_action(*this, params, state, execute);
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void OffloadGatherAction::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
