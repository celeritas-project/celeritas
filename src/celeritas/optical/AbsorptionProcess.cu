//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/AbsorptionProcess.cu
//---------------------------------------------------------------------------//
#include "AbsorptionProcess.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
void AbsorptionProcess::execute(CoreParams const& params, CoreStateDevice& state) const
{
    auto execute = make_action_track_executor(
            params.ptr<MemSpace::native>(),
            state.ptr(),
            this->action_id(),
            InteractionApplier{AbsorptionScatteringExecutor{this->host_ref()}});
    return launch_action(*this, params, state, execute);
}
//---------------------------------------------------------------------------//
}  // namespace celeritas
