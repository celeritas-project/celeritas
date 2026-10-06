//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/CoreState.cu
//---------------------------------------------------------------------------//
#include "CoreState.hh"

#include "corecel/sys/KernelLauncher.device.hh"
#include "celeritas/track/CounterExecutors.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Add to the number of pending optical photons.
 */
template<>
void CoreState<MemSpace::device>::add_pending(size_type count)
{
    AddPendingExecutor<size_type> execute_thread{
        this->ref().init.counters.data(), count};
    static KernelLauncher<decltype(execute_thread)> const launch_kernel(
        "add-pending");
    launch_kernel(1, this->stream_id(), execute_thread);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
