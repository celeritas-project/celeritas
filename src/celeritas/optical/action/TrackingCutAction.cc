//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/action/TrackingCutAction.cc
//---------------------------------------------------------------------------//
#include "TrackingCutAction.hh"

#include "corecel/io/Logger.hh"
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"

#include "ActionLauncher.hh"
#include "TrackSlotExecutor.hh"

#include "detail/TrackingCutExecutor.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Construct with action ID.
 */
TrackingCutAction::TrackingCutAction(ActionId aid)
    : StaticConcreteAction(
          aid, "tracking-cut", "kill a track and deposit its energy")
{
}

//---------------------------------------------------------------------------//
/*!
 * Launch the boundary action on host.
 */
void TrackingCutAction::step(CoreParams const& params,
                             CoreStateHost& state) const
{
    auto execute = make_action_thread_executor(params.ptr<MemSpace::native>(),
                                               state.ptr(),
                                               this->action_id(),
                                               detail::TrackingCutExecutor{});
    return launch_action(state, execute);
}

#if !CELER_USE_DEVICE
void TrackingCutAction::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
