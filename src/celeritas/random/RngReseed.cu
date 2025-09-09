//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
#if !defined(__DOXYGEN__) || __DOXYGEN__ > 0x010908
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
#endif
}  // namespace celeritas
