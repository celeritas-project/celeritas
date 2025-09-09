//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/action/TrackingCutAction.cu
//---------------------------------------------------------------------------//
#include "TrackingCutAction.hh"

#include "corecel/io/Logger.hh"
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"

#include "ActionLauncher.device.hh"
#include "TrackSlotExecutor.hh"

#include "detail/TrackingCutExecutor.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Launch the tracking cut action on device.
 */
void TrackingCutAction::step(CoreParams const& params,
                             CoreStateDevice& state) const
{
    auto execute = make_action_thread_executor(params.ptr<MemSpace::native>(),
                                               state.ptr(),
                                               this->action_id(),
                                               detail::TrackingCutExecutor{});

    static ActionLauncher<decltype(execute)> const launch_kernel(*this);
    launch_kernel(state, execute);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
