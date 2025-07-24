//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/PostBoundaryAction.cu
//---------------------------------------------------------------------------//
#include "PostBoundaryAction.hh"

#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/action/ActionLauncher.device.hh"
#include "celeritas/optical/action/TrackSlotExecutor.hh"

#include "PostBoundaryExecutor.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Launch the boundary finalization action on device.
 */
void PostBoundaryAction::step(CoreParams const& params,
                              CoreStateDevice& state) const
{
    auto execute = make_action_thread_executor(params.ptr<MemSpace::native>(),
                                               state.ptr(),
                                               this->action_id(),
                                               PostBoundaryExecutor{});

    static ActionLauncher<decltype(execute)> const launch_kernel(*this);
    launch_kernel(state, execute);
}
//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
