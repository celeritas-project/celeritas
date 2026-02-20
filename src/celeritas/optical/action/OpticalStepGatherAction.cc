//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/action/OpticalStepGatherAction.cc
//---------------------------------------------------------------------------//
#include "OpticalStepGatherAction.hh"

#include "corecel/data/AuxState.hh"
#include "corecel/data/AuxStateVec.hh"

#include "ActionLauncher.hh"
#include "TrackSlotExecutor.hh"
#include "../detail/OpticalStepGatherExecutor.hh"
namespace celeritas
{
namespace optical
{

//---------------------------------------------------------------------------//
OpticalStepGatherAction::OpticalStepGatherAction(ActionId id, SPParams params)
    : StaticConcreteAction(
          id, "optical-step-gather", "gather optical post-step data")
    , step_params_(std::move(params))
{
    CELER_EXPECT(step_params_);
}
void OpticalStepGatherAction::step(CoreParams const& params,
                                   CoreStateHost& state) const
{
    auto const& core_params = params.ref<MemSpace::native>();
    auto& core_state = state.ref();

    auto& step_state = step_params_->state_ref<MemSpace::native>(*state.aux());
    auto execute
        = TrackSlotExecutor(params.ptr<MemSpace::native>(),
                            state.ptr(),
                            detail::OpticalStepGatherExecutor<StepPoint::post>{
                                core_params, core_state, step_state});
    CELER_LOG(info) << "OpticalStepGatherAction::step State size "
                    << state.size();
    launch_action(state, execute);
}
#if !CELER_USE_DEVICE
void OpticalStepGatherAction::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
