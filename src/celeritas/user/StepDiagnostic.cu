//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/StepDiagnostic.cu
//---------------------------------------------------------------------------//
#include "StepDiagnostic.hh"

#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/LaunchAction.device.hh"
#include "celeritas/global/TrackExecutor.hh"

#include "detail/StepDiagnosticExecutor.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Execute action with device data.
 */
void StepDiagnostic::execute(CoreParams const& params,
                             CoreStateDevice& state) const
{
    auto execute = make_active_track_executor(
        params.ptr<MemSpace::native>(),
        state.ptr(),
        detail::StepDiagnosticExecutor{
            store_.params<MemSpace::native>(),
            store_.state<MemSpace::native>(state.stream_id(),
                                           this->state_size())});
    static Launcher<decltype(execute)> const launch_kernel(*this);
    launch_kernel(state, execute);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
