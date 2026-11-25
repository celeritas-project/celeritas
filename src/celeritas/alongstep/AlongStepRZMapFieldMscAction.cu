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
#include "detail/ElossApplier.hh"
#include "detail/FieldPropagationLauncher.device.hh"
#include "detail/FluctELoss.hh"
#include "detail/MeanELoss.hh"
#include "detail/MscApplier.hh"
#include "detail/MscStepLimitApplier.hh"
#include "detail/SafetyApplier.hh"
#include "detail/TimeUpdater.hh"
#include "detail/TrackUpdater.hh"

// Field classes
#include "celeritas/field/RZMapField.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Launch the along-step action on device.
 */
void AlongStepRZMapFieldMscAction::step(CoreParams const& params,
                                        CoreStateDevice& state) const
{
    if (this->has_msc()
        || field_->ref<MemSpace::native>().options.use_safety_sphere)
    {
        detail::launch_safety(*this,
                              detail::get_msc_ref<MemSpace::native>(msc_.get()),
                              params,
                              state);
    }
    if (this->has_msc())
    {
        detail::launch_limit_msc_step(
            *this, msc_->ref<MemSpace::native>(), params, state);
    }
    {
        ScopedProfiling profile_this{"propagate"};
        detail::launch_field_propagation<RZMapField>(
            *this,
            field_->ref<MemSpace::native>(),
            IsAlongStepActionEqual{this->action_id()},
            "propagate-rzmap",
            params,
            state);
    }
    if (this->has_msc())
    {
        detail::launch_scatter_safety(
            *this, msc_->ref<MemSpace::native>(), params, state);
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
