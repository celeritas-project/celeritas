//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/LocalOpticalGenOffload.cu
//---------------------------------------------------------------------------//
#include "LocalOpticalGenOffload.hh"

#include "corecel/sys/KernelLauncher.device.hh"
#include "accel/detail/UpdatePendingExecutor.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Call the UpdatePending functor to update number of primaries to be generated
 * to include the buffered optical photons; use only one device thread.
 */
void LocalOpticalGenOffload::update_primaries(
    optical::CoreParams const& optical_params,
    optical::CoreState<MemSpace::device>& state) const
{
    optical::detail::UpdatePendingExecutor execute_thread{
        optical_params.ptr<MemSpace::device>(), state.ptr(), num_photons_};
    static KernelLauncher<decltype(execute_thread)> const launch_kernel(
        "update-pending");
    launch_kernel(1, state.stream_id(), execute_thread);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
