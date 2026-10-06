//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/action/LocateVacanciesAction.cu
//---------------------------------------------------------------------------//
#include "LocateVacanciesAction.hh"

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/sys/KernelLauncher.device.hh"
#include "celeritas/track/CounterExecutors.hh"

#include "../CoreParams.hh"
#include "../CoreState.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Update the number of active slots as the empty slots have been compacted.
 */
void LocateVacanciesAction::update_alive(CoreStateDevice& state,
                                         size_type state_size) const
{
    UpdateAliveExecutor execute_thread{state.ref().init.counters.data(),
                                       state_size};
    static KernelLauncher<decltype(execute_thread)> const launch_kernel(
        "update-alive");
    launch_kernel(1, state.stream_id(), execute_thread);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
