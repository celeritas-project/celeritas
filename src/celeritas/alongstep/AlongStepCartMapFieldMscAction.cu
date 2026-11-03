//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/alongstep/AlongStepCartMapFieldMscAction.cu
//---------------------------------------------------------------------------//
#include "AlongStepCartMapFieldMscAction.hh"

#include "corecel/sys/ScopedProfiling.hh"
#include "celeritas/em/params/FluctuationParams.hh"
#include "celeritas/em/params/UrbanMscParams.hh"
#include "celeritas/field/CartMapField.hh"
#include "celeritas/field/CartMapFieldParams.hh"
#include "celeritas/global/ActionLauncher.device.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"

#include "detail/AlongStepKernels.hh"
#include "detail/FieldPropagationLauncher.device.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Launch the along-step action on device.
 */
void AlongStepCartMapFieldMscAction::step(CoreParams const& params,
                                          CoreStateDevice& state) const
{
    if (this->has_msc())
    {
        detail::launch_limit_msc_step(
            *this, msc_->ref<MemSpace::native>(), params, state);
    }
    {
        ScopedProfiling profile_this{"propagate"};
        detail::launch_field_propagation<CartMapField>(
            *this,
            "propagate-cartmap",
            IsAlongStepActionEqual{this->action_id()},
            field_->ref<MemSpace::native>(),
            field_diag_.get(),
            params,
            state);
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
