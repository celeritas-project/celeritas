//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/alongstep/AlongStepNeutralAction.cu
//---------------------------------------------------------------------------//
#include "AlongStepNeutralAction.hh"

#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/LaunchAction.device.hh"
#include "celeritas/global/TrackExecutor.hh"

#include "detail/AlongStepNeutral.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Launch the along-step action on device.
 */
void AlongStepNeutralAction::execute(CoreParams const& params,
                                     CoreStateDevice& state) const
{
    auto execute
        = make_along_step_track_executor(params.ptr<MemSpace::native>(),
                                         state.ptr(),
                                         this->action_id(),
                                         detail::AlongStepNeutralExecutor{});
    static Launcher<decltype(execute)> const launch_kernel(*this);
    launch_kernel(state, execute);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
