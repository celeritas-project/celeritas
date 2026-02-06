//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/detail/SetPending.cu
//---------------------------------------------------------------------------//
#include "SetPending.hh"

#include "corecel/Assert.hh"
#include "corecel/Types.hh"

#include "../ActionLauncher.device.hh"
#include "../CoreParams.hh"
#include "../CoreState.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Reset the num_pending counter to the number of generated primaries.
 */
void set_pending(CoreParams const& params,
                 CoreState<MemSpace::device>& state,
                 size_type num_primaries)
{
    SetPendingExecutor execute_thread{
        params.ptr<MemSpace::native>(), state.ptr(), num_primaries};
    static ActionLauncher<decltype(execute_thread)> const launch_kernel(
        "set-pending");
    launch_kernel(1, state, execute_thread);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
