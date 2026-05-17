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
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/TrackExecutor.hh"

#include "detail/UpdateAliveExecutor.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Update the number of active slots as the empty slots have been compacted.
 */
void LocateVacanciesAction::update_alive(CoreParams const& params,
                                         CoreStateDevice& state,
                                         size_type state_size) const
{
    auto execute_thread
        = make_single_track_executor(params.ptr<MemSpace::native>(),
                                     state.ptr(),
                                     detail::UpdateAliveExecutor{state_size});
    static KernelLauncher<decltype(execute_thread)> const launch_kernel(
        "update-alive");
    launch_kernel(1, state.stream_id(), execute_thread);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
