//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/GeneratorBase.cu
//---------------------------------------------------------------------------//
#include "GeneratorBase.hh"

#include "corecel/Assert.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/TrackExecutor.hh"
#include "celeritas/optical/action/ActionLauncher.device.hh"

#include "detail/UpdatePendingExecutor.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Launch a (device) kernel to update the number of pending optical photons.
 */
void GeneratorBase::update_pending(CoreParams const& params,
                                   CoreStateDevice& state,
                                   size_type num_pending) const
{
    // Update the number of pending optical photons
    auto execute_thread = make_single_track_executor(
        params.ptr<MemSpace::native>(),
        state.ptr(),
        detail::UpdatePendingExecutor{num_pending});
    static KernelLauncher<decltype(execute_thread)> const launch_kernel(
        "update-pending");
    launch_kernel(1, state.stream_id(), execute_thread);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
