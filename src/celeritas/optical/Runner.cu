//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/Runner.cu
//---------------------------------------------------------------------------//
#include "Runner.hh"

#include "corecel/Assert.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/action/ActionLauncher.device.hh"

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
    detail::UpdatePendingExecutor execute{state.ptr(), num_pending};
    static KernelLauncher<decltype(execute)> const launch_kernel(
        "update-pending");
    launch_kernel(1, state.stream_id(), execute);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
