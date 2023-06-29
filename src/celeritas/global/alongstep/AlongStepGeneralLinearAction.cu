//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/alongstep/AlongStepGeneralLinearAction.cu
//---------------------------------------------------------------------------//
#include "AlongStepGeneralLinearAction.hh"

#include "corecel/sys/ScopedProfiling.hh"
#include "celeritas/em/FluctuationParams.hh"  // IWYU pragma: keep
#include "celeritas/em/UrbanMscParams.hh"  // IWYU pragma: keep

#include "detail/AlongStepKernels.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Launch the along-step action on device.
 *
 * The six kernels should correspons to the six function calls in \c AlongStep.
 */
void AlongStepGeneralLinearAction::execute(CoreParams const& params,
                                           CoreStateDevice& state) const
{
    ScopedProfiling profile_this{label()};
    if (this->has_msc())
    {
        ScopedProfiling profile_this{label() + "-limit-msc-step"};
        detail::launch_limit_msc_step(
            *this, msc_->ref<MemSpace::native>(), params, state);
    }
    {
        ScopedProfiling profile_this{label() + "-propagate"};
        detail::launch_propagate(*this, params, state);
    }
    if (this->has_msc())
    {
        ScopedProfiling profile_this{label() + "-apply-msc"};
        detail::launch_apply_msc(
            *this, msc_->ref<MemSpace::native>(), params, state);
    }
    {
        ScopedProfiling profile_this{label() + "-update-time"};
        detail::launch_update_time(*this, params, state);
    }
    if (this->has_fluct())
    {
        ScopedProfiling profile_this{label() + "-apply-eloss"};
        detail::launch_apply_eloss(
            *this, fluct_->ref<MemSpace::native>(), params, state);
    }
    else
    {
        ScopedProfiling profile_this{label() + "-apply-eloss"};
        detail::launch_apply_eloss(*this, params, state);
    }
    {
        ScopedProfiling profile_this{label() + "-update-track"};
        detail::launch_update_track(*this, params, state);
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
