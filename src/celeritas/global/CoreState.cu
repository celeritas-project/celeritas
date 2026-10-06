//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/CoreState.cu
//---------------------------------------------------------------------------//
#include "CoreState.hh"

#include "corecel/Assert.hh"
#include "corecel/sys/KernelLauncher.device.hh"
#include "celeritas/track/CounterExecutors.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Reset counters that are accumulated during a step.
 */
template<>
void CoreState<MemSpace::device>::reset_counters()
{
    ResetCountersExecutor execute_thread{this->ref().init.counters.data()};
    static KernelLauncher<decltype(execute_thread)> const launch_kernel(
        "reset_counters");
    launch_kernel(1, this->stream_id(), execute_thread);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
