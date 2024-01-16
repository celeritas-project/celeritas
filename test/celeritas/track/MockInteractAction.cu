//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/MockInteractAction.cu
//---------------------------------------------------------------------------//
#include "MockInteractAction.hh"

#include "celeritas/global/ActionLauncher.device.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"

#include "MockInteractExecutor.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
void MockInteractAction::execute(CoreParams const& params,
                                 CoreStateDevice& state) const
{
    auto execute = make_active_track_executor(
        params.ptr<MemSpace::native>(),
        state.ptr(),
        MockInteractExecutor{data_.ref<MemSpace::native>()});
    static ActionLauncher<decltype(execute)> const launch_kernel(*this);
    launch_kernel(state, execute);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
