//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/detail/SetGenerated.cc
//---------------------------------------------------------------------------//
#include "corecel/Assert.hh"
#include "corecel/Types.hh"

#include "SetGeneratedExecutor.hh"
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
void set_generated(CoreParams const& params, CoreState<MemSpace::host>& state)
{
    SetGeneratedExecutor execute_thread{params.ptr<MemSpace::native>(),
                                        state.ptr()};
    launch_core(1, "set-generated", params, state, execute_thread);
}

//---------------------------------------------------------------------------//
// DEVICE-DISABLED IMPLEMENTATION
//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void set_generated(CoreParams const&, CoreState<MemSpace::device>&)
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
