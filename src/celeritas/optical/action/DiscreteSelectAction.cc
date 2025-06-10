//------------------------------ -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/action/DiscreteSelectAction.cc
//---------------------------------------------------------------------------//
#include "DiscreteSelectAction.hh"

#include "corecel/io/Logger.hh"
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"

#include "ActionLauncher.hh"
#include "DiscreteSelectExecutor.hh"
#include "TrackSlotExecutor.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Construct with an action ID.
 */
DiscreteSelectAction::DiscreteSelectAction(ActionId id)
    : StaticConcreteAction(id,
                           "optical-discrete-select",
                           "select a discrete optical interaction")
{
}

//---------------------------------------------------------------------------//
/*!
 * Launch the discrete-select action on host.
 */
void DiscreteSelectAction::step(CoreParams const& params,
                                CoreStateHost& state) const
{
    launch_action(state,
                  make_action_thread_executor(params.ptr<MemSpace::native>(),
                                              state.ptr(),
                                              this->action_id(),
                                              DiscreteSelectExecutor{}));
}

//---------------------------------------------------------------------------//
/*!
 * Launch the discrete-select action on device.
 */
#if !CELER_USE_DEVICE
void DiscreteSelectAction::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
