//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/ExtendFromPrimariesAction.cu
//---------------------------------------------------------------------------//
#include "ExtendFromPrimariesAction.hh"

#include "celeritas/global/ActionLauncher.device.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"

#include "detail/ProcessPrimariesExecutor.hh"
#include "detail/UpdateCountersExecutor.hh"

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
        params.ptr<MemSpace::native>(), state.ptr(), primaries, pstate.count};
    static ActionLauncher<decltype(execute_thread)> const launch_kernel(*this);
    if (!primaries.empty())
    {
        launch_kernel(primaries.size(), state.stream_id(), execute_thread);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Launch a (device) kernel to update state counters based on the number of
 * primary particles.
 */
void ExtendFromPrimariesAction::update_counters(CoreParams const& params,
                                                CoreStateDevice& state,
                                                size_type num_primaries) const
{
    auto execute_thread = make_single_track_executor(
        params.ptr<MemSpace::native>(),
        state.ptr(),
        detail::UpdateCountersExecutor{num_primaries});
    static KernelLauncher<decltype(execute_thread)> const launch_kernel(
        "update-counters");
    launch_kernel(1, state.stream_id(), execute_thread);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
