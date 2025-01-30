//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/action/DiscreteSelectAction.cu
//---------------------------------------------------------------------------//
#include "DiscreteSelectAction.hh"

#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"

#include "ActionLauncher.device.hh"
#include "DiscreteSelectExecutor.hh"
#include "TrackSlotExecutor.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Launch the discrete-select action on device.
 */
void DiscreteSelectAction::step(CoreParams const& core_params,
                                CoreStateDevice& core_state) const
{
    auto execute
        = make_action_thread_executor(core_params.ptr<MemSpace::native>(),
                                      core_state.ptr(),
                                      this->action_id(),
                                      DiscreteSelectExecutor{});
    static ActionLauncher<decltype(execute)> const launch_kernel(*this);
    launch_kernel(core_state, execute);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
