//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/alongstep/AlongStepRZMapFieldMscAction.cu
//---------------------------------------------------------------------------//
#include "AlongStepRZMapFieldMscAction.hh"

#include "corecel/sys/ScopedProfiling.hh"
#include "celeritas/em/params/FluctuationParams.hh"
#include "celeritas/em/params/UrbanMscParams.hh"
#include "celeritas/field/RZMapFieldParams.hh"
#include "celeritas/global/ActionLauncher.device.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"

#include "detail/AlongStepKernels.hh"
#include "detail/PropagationApplier.hh"
#include "detail/RZMapFieldPropagatorFactory.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Launch the along-step action on device.
 */
void AlongStepRZMapFieldMscAction::step(CoreParams const& params,
                                        CoreStateDevice& state) const
{
    if (this->has_msc())
    {
        detail::launch_limit_msc_step(
            *this, msc_->ref<MemSpace::native>(), params, state);
    }
    {
        ScopedProfiling profile_this{"propagate"};
        auto execute_thread = make_along_step_track_executor(
            params.ptr<MemSpace::native>(),
            state.ptr(),
            this->action_id(),
            detail::PropagationApplier{detail::RZMapFieldPropagatorFactory{
                field_->ref<MemSpace::native>()}});
        static ActionLauncher<decltype(execute_thread)> const launch_kernel(
            *this, "propagate-rzmap");
        launch_kernel(*this, params, state, execute_thread);
    }
    if (this->has_msc())
    {
        detail::launch_apply_msc(
            *this, msc_->ref<MemSpace::native>(), params, state);
    }
    detail::launch_update_time(*this, params, state);
    if (this->has_fluct())
    {
        detail::launch_apply_eloss(
            *this, fluct_->ref<MemSpace::native>(), params, state);
    }
    else
    {
        detail::launch_apply_eloss(*this, params, state);
    }
    detail::launch_update_track(*this, params, state);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
