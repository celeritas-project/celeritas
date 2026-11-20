//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/HandBackTestAction.cu
//---------------------------------------------------------------------------//
#include "HandBackTestAction.hh"

#include "celeritas/global/ActionLauncher.device.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Mark tracks on device.
 */
void HandBackTestAction::step(CoreParams const& params,
                              CoreStateDevice& state) const
{
    auto execute = make_active_track_executor(params.ptr<MemSpace::native>(),
                                              state.ptr(),
                                              HandBackTestExecutor{trigger_});
    static ActionLauncher<decltype(execute)> const launch_kernel(*this);
    launch_kernel(state, execute);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
