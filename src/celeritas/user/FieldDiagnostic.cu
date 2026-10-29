//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/FieldDiagnostic.cu
//---------------------------------------------------------------------------//
#include "FieldDiagnostic.hh"

#include "celeritas/global/ActionLauncher.device.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"

#include "detail/FieldDiagnosticExecutor.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Bin the propagation records with device data.
 */
void FieldDiagnostic::step(CoreParams const& params,
                           CoreStateDevice& state) const
{
    auto& diag_state
        = this->state<MemSpace::native>(state.stream_id(), state.size());
    auto execute = make_active_track_executor(
        params.ptr<MemSpace::native>(),
        state.ptr(),
        detail::FieldDiagnosticExecutor{this->params<MemSpace::native>(),
                                        diag_state});
    static ActionLauncher<decltype(execute)> const launch_kernel(*this);
    launch_kernel(state, execute);
    ++diag_state.step_iter;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
