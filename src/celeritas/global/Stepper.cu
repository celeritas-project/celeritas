//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/Stepper.cu
//---------------------------------------------------------------------------//
#include "Stepper.hh"

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/sys/KernelLauncher.device.hh"
#include "celeritas/track/CounterExecutors.hh"

#include "CoreParams.hh"
#include "CoreState.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Set the num_pending counter to the number of generated primaries.
 */
template<>
void Stepper<MemSpace::device>::reset_counters()
{
    ResetCountersExecutor execute_thread{state_->ref().init.counters.data()};
    static KernelLauncher<decltype(execute_thread)> const launch_kernel(
        "reset_counters");
    launch_kernel(1, state_->stream_id(), execute_thread);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
