//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/AbsorptionProcess.cc
//---------------------------------------------------------------------------//
#include "AbsorptionProcess.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//

AbsorptionProcess::AbsorptionProcess()
    : OpticalProcess(action_id)
{
}

auto AbsorptionProcess::step_limits() const
{

}

void AbsorptionProcess::execute(CoreParams const& params, CoreStateHost& state) const
{
    auto execute = make_action_track_executor(
            params.ptr<MemSpace::native>(),
            state.ptr(),
            this->action_id(),
            InteractionApplier{AbsorptionScatteringExecutor{this->host_ref()}});
    return launch_action(*this, params, state, execute);
}

#if !CELER_USE_DEVICE
void AbsorptionProcess::execute(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
