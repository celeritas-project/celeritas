//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/LocalOpticalGenOffload.cu
//---------------------------------------------------------------------------//
#include "LocalOpticalGenOffload.hh"

#include "corecel/sys/KernelLauncher.device.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/optical/Transporter.hh"
#include "celeritas/track/CounterExecutors.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Call the UpdatePending functor to update number of primaries to be generated
 * to include the buffered optical photons; use only one device thread.
 */
void LocalOpticalGenOffload::update_primaries(
    optical::CoreState<MemSpace::device>& state) const
{
    auto const& optical_params = *transport_->params();

    AddPendingExecutor<size_type> execute_thread{
        state.ref().init.counters.data(), num_photons_};

    static KernelLauncher<decltype(execute_thread)> const launch_kernel(
        "update-pending");
    launch_kernel(1, state.stream_id(), execute_thread);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
