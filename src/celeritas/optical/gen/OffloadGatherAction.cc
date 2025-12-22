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

#include "detail/OffloadPreGatherExecutor.hh"
#include "detail/OffloadPrePostGatherExecutor.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct and add to core params.
 */
template<StepActionOrder S>
std::shared_ptr<OffloadGatherAction<S>>
OffloadGatherAction<S>::make_and_insert(CoreParams const& core)
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
template<StepActionOrder S>
OffloadGatherAction<S>::OffloadGatherAction(ActionId action_id, AuxId aux_id)
    : action_id_(action_id), aux_id_(aux_id)
{
    CELER_EXPECT(action_id_);
    CELER_EXPECT(aux_id_);
}

//---------------------------------------------------------------------------//
/*!
 * Build state data for a stream.
 */
template<StepActionOrder S>
auto OffloadGatherAction<S>::create_state(MemSpace m,
                                          StreamId id,
                                          size_type size) const -> UPState
{
    return make_aux_state<TraitsT::template Data>(m, id, size);
}

//---------------------------------------------------------------------------//
/*!
 * Gather pre-step data.
 */
template<StepActionOrder S>
void OffloadGatherAction<S>::step(CoreParams const& params,
                                  CoreStateHost& state) const
{
    auto& step = state.aux_data<TraitsT::template Data>(aux_id_);
    auto execute = make_active_track_executor(
        params.ptr<MemSpace::native>(), state.ptr(), Executor{step});
    launch_action(*this, params, state, execute);
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
template<StepActionOrder S>
void OffloadGatherAction<S>::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//

template class OffloadGatherAction<StepActionOrder::pre>;
template class OffloadGatherAction<StepActionOrder::pre_post>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
