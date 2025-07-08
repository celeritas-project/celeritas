//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/InitBoundaryAction.cc
//---------------------------------------------------------------------------//
#include "InitBoundaryAction.hh"

#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/action/ActionLauncher.hh"
#include "celeritas/optical/action/TrackSlotExecutor.hh"

#include "InitBoundaryExecutor.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Construct the boundary crossing initialization action from an action ID.
 */
InitBoundaryAction::InitBoundaryAction(ActionId aid)
    : ConcreteAction(aid,
                     "optical-boundary-init",
                     "Initialize optical boundary crossing action")
{
}

//---------------------------------------------------------------------------//
/*!
 * Execute the model on the host.
 */
void InitBoundaryAction::step(CoreParams const& params,
                              CoreStateHost& state) const
{
    launch_action(state,
                  make_action_thread_executor(params.ptr<MemSpace::native>(),
                                              state.ptr(),
                                              this->action_id(),
                                              InitBoundaryExecutor{}));
}

//---------------------------------------------------------------------------//
/*!
 * Execute the model on the device.
 */
#if !CELER_USE_DEVICE
void InitBoundaryAction::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
