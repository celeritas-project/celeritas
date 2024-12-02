//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/random/RngReseed.cu
//---------------------------------------------------------------------------//
#include "RngReseed.hh"

#include "corecel/Types.hh"
#include "corecel/sys/KernelLauncher.device.hh"

#include "detail/RngReseedExecutor.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// KERNEL INTERFACE
//---------------------------------------------------------------------------//
/*!
 * Reinitialize the RNG states on device at the start of an event.
 *
 * Each thread's state is initialized using same seed and skipped ahead a
 * different number of subsequences so the sequences on different threads will
 * not have statistically correlated values.
 */
void reseed_rng(DeviceCRef<RngParamsData> const& params,
                DeviceRef<RngStateData> const& state,
                StreamId stream,
                UniqueEventId event_id)
{
    detail::RngReseedExecutor execute_thread{params, state, event_id};
    static KernelLauncher<decltype(execute_thread)> const launch_kernel(
        "rng-reseed");
    launch_kernel(state.size(), stream, execute_thread);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
