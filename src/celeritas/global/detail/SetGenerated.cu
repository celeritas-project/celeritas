//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/detail/SetGenerated.cu
//---------------------------------------------------------------------------//
#include "SetGenerated.hh"

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
void set_generated(CoreParams const& params, CoreState<MemSpace::device>& state)
{
    SetGeneratedExecutor execute_thread{params.ptr<MemSpace::native>(),
                                        state.ptr()};
    static ActionLauncher<decltype(execute_thread)> const launch_kernel(
        "set-generated");
    launch_kernel(1, state, execute_thread);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
