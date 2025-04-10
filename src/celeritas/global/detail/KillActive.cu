//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/detail/KillActive.cu
//---------------------------------------------------------------------------//
#include "KillActive.hh"

#include "corecel/Assert.hh"
#include "corecel/Types.hh"

#include "../ActionLauncher.device.hh"
#include "../CoreParams.hh"
#include "../CoreState.hh"
#include "../TrackExecutor.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Mark all active tracks as "errored".
 */
void kill_active(CoreParams const& params, CoreState<MemSpace::device>& state)
{
    TrackExecutor execute_thread{
        params.ptr<MemSpace::native>(), state.ptr(), KillActiveExecutor{}};
    static ActionLauncher<decltype(execute_thread)> const launch_kernel(
        "kill-active");
    return launch_kernel(state, execute_thread);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
