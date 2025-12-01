//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/data/detail/RanluxppRngStateInit.cu
//---------------------------------------------------------------------------//
#include "RanluxppRngStateInit.hh"

#include "corecel/Assert.hh"
#include "corecel/sys/KernelLauncher.device.hh"
#include "corecel/sys/ThreadId.hh"

namespace celeritas
{
namespace detail
{

//---------------------------------------------------------------------------//
/*!
 * Initialize the Ranlux states on device from user provided seed on host.
 */
void ranlux_state_init(DeviceCRef<RanluxppRngParamsData> const& params,
                       DeviceRef<RanluxppRngStateData> const& state,
                       StreamId stream)
{
    CELER_EXPECT(stream);
    detail::RanluxppRngSeedExecutor execute_thread{params, state};
    static KernelLauncher<decltype(execute_thread)> const launch_kernel(
        "rng-reseed");
    launch_kernel(state.size(), stream, execute_thread);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
