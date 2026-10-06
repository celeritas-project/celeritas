//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/Runner.cu
//---------------------------------------------------------------------------//
#include "Runner.hh"

#include "corecel/Assert.hh"
#include "corecel/sys/KernelLauncher.device.hh"
#include "celeritas/track/CounterExecutors.hh"

#include "CoreState.hh"

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
    AddPendingExecutor<size_type> execute_thread{
        state.ref().init.counters.data(), num_pending};
    static KernelLauncher<decltype(execute_thread)> const launch_kernel(
        "update-pending");
    launch_kernel(1, state.stream_id(), execute_thread);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
