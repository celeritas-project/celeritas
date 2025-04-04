//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/detail/KillActive.cc
//---------------------------------------------------------------------------//
#include "KillActive.hh"

#include "corecel/Assert.hh"
#include "corecel/Types.hh"

#include "../ActionLauncher.hh"
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
void kill_active(CoreParams const& params, CoreState<MemSpace::host>& state)
{
    TrackExecutor execute_thread{
        params.ptr<MemSpace::native>(), state.ptr(), KillActiveExecutor{}};
    return launch_core("kill-active", params, state, execute_thread);
}

#if !CELER_USE_DEVICE
void kill_active(CoreParams const&, CoreState<MemSpace::device>&)
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
