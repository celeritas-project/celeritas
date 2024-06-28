//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/AbsorptionModel.cc
//---------------------------------------------------------------------------//
#include "AbsorptionModel.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
AbsorptionModel::AbsorptionModel(ActionId id, SPConstImported imported)
    : OpticalModel(id, "absorption", "interact by optical absorption")
    , imported_(imported)
{}

auto AbsorptionModel::step_limits(OpticalMaterialId opt_mat) const -> StepLimitBuilder
{
    return imported_.step_limits(opt_mat);
}

void AbsorptionModel::execute(OpticalParams const& params, OpticalStateHost& state) const
{
    // TODO: Need an optical state version of make action track executor and launch action?
    auto execute = make_action_track_executor(
            params.ptr<MemSpace::native>(),
            state.ptr(),
            this->action_id(),
            InteractionApplier{AbsorptionExecutor{}});
    return launch_action(*this, params, state, execute);
}

#if !CELER_USE_DEVICE
void AbsorptionModel::execute(OpticalParams const&, OpticalStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
