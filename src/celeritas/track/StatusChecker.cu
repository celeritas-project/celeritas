//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/StatusChecker.cu
//---------------------------------------------------------------------------//
#include "StatusChecker.hh"

#include "corecel/device_runtime_api.hh"

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
void StatusChecker::launch_impl(
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
    CELER_DEVICE_CALL_PREFIX(StreamSynchronize(
        celeritas::device().stream(state.stream_id()).get()));
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
