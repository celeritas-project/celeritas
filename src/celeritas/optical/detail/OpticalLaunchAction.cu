//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detail/OpticalLaunchAction.cu
//---------------------------------------------------------------------------//
#include "OpticalLaunchAction.hh"

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/sys/KernelLauncher.device.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"
#include "celeritas/optical/action/ActionLauncher.device.hh"

#include "CheckPendingExecutor.hh"
#include "../CoreParams.hh"
#include "../CoreState.hh"
#include "../TrackExecutor.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Determine whether to transport the pending optical tracks.
 */
void OpticalLaunchAction::confirm_transport(CoreParams const& params,
                                            CoreStateDevice& core_state) const
{
    auto& optical_state = get<optical::CoreState<MemSpace::native>>(
        core_state.aux(), this->aux_id());
    // Default is to reset flag to true for this iteration
    bool result{true};
    auto execute_thread = make_single_track_executor(
        params.ptr<MemSpace::native>(),
        core_state.ptr(),
        CheckPendingExecutor{optical_state.ptr(),
                             core_state.stream_id(),
                             data_.auto_flush,
                             &result});
    static KernelLauncher<decltype(execute_thread)> const launch_kernel(
        "check-pending");
    launch_kernel(1, core_state.stream_id(), execute_thread);
    transport_tracks_ = result;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
