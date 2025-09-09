//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/detail/TrackingCutAction.cu
//---------------------------------------------------------------------------//
#include "TrackingCutAction.hh"

#include "celeritas/global/ActionLauncher.device.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"

#include "TrackingCutExecutor.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Launch the action on device.
 */
void TrackingCutAction::step(CoreParams const& params,
                             CoreStateDevice& state) const
{
    auto execute = make_action_track_executor(params.ptr<MemSpace::native>(),
                                              state.ptr(),
                                              this->action_id(),
                                              TrackingCutExecutor{});

    static ActionLauncher<decltype(execute)> const launch_kernel(*this);
    launch_kernel(*this, params, state, execute);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
