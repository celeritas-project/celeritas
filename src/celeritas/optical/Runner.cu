//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/Runner.cu
//---------------------------------------------------------------------------//
#include "Runner.hh"

#include "corecel/Assert.hh"

#include "CoreParams.hh"
#include "CoreState.hh"
#include "TrackExecutor.hh"
#include "action/ActionLauncher.device.hh"
#include "gen/detail/UpdatePendingExecutor.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Launch a (device) kernel to update the number of pending optical photons.
 */
void Runner::update_pending(CoreState<MemSpace::device>& state,
                            size_type num_pending) const
{
    // Update the number of pending optical photons
    auto execute_thread = make_single_track_executor(
        this->params()->ptr<MemSpace::native>(),
        state.ptr(),
        detail::UpdatePendingExecutor{num_pending});
    static KernelLauncher<decltype(execute_thread)> const launch_kernel(
        "update-pending");
    launch_kernel(1, state.stream_id(), execute_thread);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
