//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
void DiscreteSelectAction::step(CoreParams const& params,
                                CoreStateDevice& state) const
{
    auto execute = make_action_thread_executor(params.ptr<MemSpace::native>(),
                                               state.ptr(),
                                               this->action_id(),
                                               DiscreteSelectExecutor{});
    static ActionLauncher<decltype(execute)> const launch_kernel(*this);
    launch_kernel(state, execute);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
