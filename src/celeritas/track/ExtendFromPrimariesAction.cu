//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/ExtendFromPrimariesAction.cu
//---------------------------------------------------------------------------//
#include "ExtendFromPrimariesAction.hh"

#include "corecel/math/Algorithms.hh"
#include "celeritas/global/ActionLauncher.device.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"

#include "detail/ProcessPrimariesExecutor.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Launch a kernel to create track initializers from primary particles.
 */
void ExtendFromPrimariesAction::process_primaries(
    CoreParams const& params,
    CoreStateDevice& state,
    PrimaryStateData<MemSpace::device> const& pstate) const
{
    auto primaries = pstate.primaries();
    detail::ProcessPrimariesExecutor execute_thread{
        params.ptr<MemSpace::native>(), state.ptr(), primaries};
    static ActionLauncher<decltype(execute_thread)> const launch_kernel(*this);
    if (!primaries.empty())
    {
        auto num_threads = max<size_type>(primaries.size(), state.size());
        launch_kernel(num_threads, state.stream_id(), execute_thread);
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
