//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//! \file celeritas/user/ActionDiagnostic.cu
//---------------------------------------------------------------------------//
#include "ActionDiagnostic.hh"

#include "celeritas/global/ActionLauncher.device.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"

#include "detail/ActionDiagnosticExecutor.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Execute action with device data.
 */
void ActionDiagnostic::step(CoreParams const& params,
                            CoreStateDevice& state) const
{
    auto execute = make_active_track_executor(
        params.ptr<MemSpace::native>(),
        state.ptr(),
        detail::ActionDiagnosticExecutor{
            store_.params<MemSpace::native>(),
            store_.state<MemSpace::native>(state.stream_id(),
                                           this->state_size())});
    static ActionLauncher<decltype(execute)> const launch_kernel(*this);
    launch_kernel(state, execute);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
