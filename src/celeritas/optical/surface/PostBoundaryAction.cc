//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/PostBoundaryAction.cc
//---------------------------------------------------------------------------//
#include "PostBoundaryAction.hh"

#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/action/ActionLauncher.hh"
#include "celeritas/optical/action/TrackSlotExecutor.hh"

#include "PostBoundaryExecutor.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Construct the post boundary crossing action from an action ID.
 */
PostBoundaryAction::PostBoundaryAction(ActionId aid)
    : ConcreteAction(aid,
                     "optical-boundary-post",
                     "Finalize optical boundary crossing action")
{
}

//---------------------------------------------------------------------------//
/*!
 * Execute the action on the host.
 */
void PostBoundaryAction::step(CoreParams const& params,
                              CoreStateHost& state) const
{
    launch_action(state,
                  make_action_thread_executor(params.ptr<MemSpace::native>(),
                                              state.ptr(),
                                              this->action_id(),
                                              PostBoundaryExecutor{}));
}

//---------------------------------------------------------------------------//
/*!
 * Execute the action on the device.
 */
#if !CELER_USE_DEVICE
void PostBoundaryAction::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
