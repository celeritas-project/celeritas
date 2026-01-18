//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/alongstep/AlongStepUniformMscAction.cu
//---------------------------------------------------------------------------//
#include "AlongStepUniformMscAction.hh"

#include "corecel/sys/ScopedProfiling.hh"
#include "celeritas/em/params/FluctuationParams.hh"
#include "celeritas/em/params/UrbanMscParams.hh"
#include "celeritas/field/DormandPrinceIntegrator.hh"
#include "celeritas/field/FieldDriverOptions.hh"
#include "celeritas/field/MakeMagFieldPropagator.hh"
#include "celeritas/field/UniformField.hh"
#include "celeritas/global/ActionLauncher.device.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"

#include "detail/AlongStepKernels.hh"
#include "detail/FieldFunctors.hh"
#include "detail/FieldTrackPropagator.hh"
#include "detail/LinearTrackPropagator.hh"
#include "detail/PropagationApplier.hh"

// Field classes
#include "celeritas/field/UniformField.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Launch the along-step action on device.
 */
void AlongStepUniformMscAction::step(CoreParams const& params,
                                     CoreStateDevice& state) const
{
    if (this->has_msc())
    {
        detail::launch_limit_msc_step(
            *this, msc_->ref<MemSpace::native>(), params, state);
    }
    auto field = field_->ref<MemSpace::native>();
    {
        ScopedProfiling profile_this{"propagate-uniform"};
        auto execute_thread = ConditionalTrackExecutor{
            params.ptr<MemSpace::native>(),
            state.ptr(),
            detail::IsAlongStepUniformField{this->action_id(), field},
            detail::PropagationApplier{
                detail::FieldTrackPropagator<UniformField>{field}}};
        static ActionLauncher<decltype(execute_thread)> const launch_kernel(
            *this, "propagate");
        launch_kernel(*this, params, state, execute_thread);
    }
    if (!field_->in_all_volumes())
    {
        // Launch linear propagation kernel for tracks in volumes without field
        ScopedProfiling profile_this{"propagate-linear"};
        auto execute_thread = ConditionalTrackExecutor{
            params.ptr<MemSpace::native>(),
            state.ptr(),
            detail::IsAlongStepLinear{this->action_id(), field},
            detail::PropagationApplier{detail::LinearTrackPropagator{}}};
        static ActionLauncher<decltype(execute_thread)> const launch_kernel(
            *this, "propagate-linear");
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
