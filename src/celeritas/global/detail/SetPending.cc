//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/detail/SetPending.cc
//---------------------------------------------------------------------------//
#include "SetPending.hh"

#include "corecel/Assert.hh"
#include "corecel/Types.hh"

#include "../ActionLauncher.hh"
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
                 CoreState<MemSpace::host>& state,
                 size_type num_primaries)
{
    SetPendingExecutor execute_thread{
        params.ptr<MemSpace::native>(), state.ptr(), num_primaries};
    launch_core(1, "set-pending", params, state, execute_thread);
}

//---------------------------------------------------------------------------//
// DEVICE-DISABLED IMPLEMENTATION
//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void set_pending(CoreParams const&, CoreState<MemSpace::device>&, size_type)
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
