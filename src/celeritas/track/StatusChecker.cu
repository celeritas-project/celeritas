//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/StatusChecker.cu
//---------------------------------------------------------------------------//
#include "StatusChecker.hh"

#include "corecel/DeviceRuntimeApi.hh"

#include "corecel/Assert.hh"
#include "corecel/sys/Stream.hh"
#include "celeritas/global/ActionLauncher.device.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"

#include "detail/StatusCheckExecutor.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Execute with with the last action's ID and the state.
 */
void StatusChecker::step_impl(
    CoreParams const& params,
    CoreState<MemSpace::device>& state,
    StatusStateRef<MemSpace::device> const& aux_state) const
{
    TrackExecutor execute_thread{
        params.ptr<MemSpace::native>(),
        state.ptr(),
        detail::StatusCheckExecutor{this->ref<MemSpace::native>(), aux_state}};
    static ActionLauncher<decltype(execute_thread)> const launch_kernel(
        this->label());
    launch_kernel(state, execute_thread);
    CELER_DEVICE_API_CALL(StreamSynchronize(
        celeritas::device().stream(state.stream_id()).get()));
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
