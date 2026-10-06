//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/GeneratorBase.cu
//---------------------------------------------------------------------------//
#include "GeneratorBase.hh"

#include "corecel/Assert.hh"
#include "corecel/sys/KernelLauncher.device.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/track/CounterExecutors.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Launch a (device) kernel to update the number of pending optical photons.
 */
void GeneratorBase::update_pending(CoreStateDevice& state,
                                   size_type num_pending) const
{
    // Update the number of pending optical photons
    UpdatePendingExecutor<size_type> execute_thread{
        state.ref().init.counters.data(), num_pending};
    static KernelLauncher<decltype(execute_thread)> const launch_kernel(
        "update-pending");
    launch_kernel(1, state.stream_id(), execute_thread);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
