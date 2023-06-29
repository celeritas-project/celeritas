//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/alongstep/AlongStepRZMapFieldMscAction.cu
//---------------------------------------------------------------------------//
#include "AlongStepRZMapFieldMscAction.hh"

#include "corecel/sys/ScopedProfiling.hh"
#include "celeritas/em/FluctuationParams.hh"
#include "celeritas/em/UrbanMscParams.hh"
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
void AlongStepRZMapFieldMscAction::execute(CoreParams const& params,
                                           CoreStateDevice& state) const
{
    ScopedProfiling profile_this{label()};
    {
        ScopedProfiling profile_this{label() + "-limit-msc-step"};
        detail::launch_limit_msc_step(
            *this, msc_->ref<MemSpace::native>(), params, state);
    }
    {
        ScopedProfiling profile_this{label() + "-propagate"};
        auto execute_thread = make_along_step_track_executor(
            params.ptr<MemSpace::native>(),
            state.ptr(),
            this->action_id(),
            detail::PropagationApplier{detail::RZMapFieldPropagatorFactory{
                field_->ref<MemSpace::native>()}});
        static ActionLauncher<decltype(execute_thread)> const launch_kernel(
            *this, "propagate-rzmap");
        launch_kernel(params, state, *this, execute_thread);
    }
    {
        ScopedProfiling profile_this{label() + "-apply-msc"};
        detail::launch_apply_msc(
            *this, msc_->ref<MemSpace::native>(), params, state);
    }
    {
        ScopedProfiling profile_this{label() + "-update-time"};
        detail::launch_update_time(*this, params, state);
    }
    {
        ScopedProfiling profile_this{label() + "-apply-eloss"};
        detail::launch_apply_eloss(
            *this, fluct_->ref<MemSpace::native>(), params, state);
    }
    {
        ScopedProfiling profile_this{label() + "-update-track"};
        detail::launch_update_track(*this, params, state);
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
