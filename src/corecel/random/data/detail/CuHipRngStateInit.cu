//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/data/detail/CuHipRngStateInit.cu
//---------------------------------------------------------------------------//
#include "CuHipRngStateInit.hh"

#include "corecel/Assert.hh"
#include "corecel/sys/KernelLauncher.device.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
// KERNEL INTERFACE
//---------------------------------------------------------------------------//
/*!
 * Initialize the RNG states on device from seeds randomly generated on host.
 */
void rng_state_init(DeviceCRef<CuHipRngParamsData> const& params,
                    DeviceRef<CuHipRngStateData> const& state,
                    DeviceCRef<CuHipRngInitData> const& seeds,
                    StreamId stream)
{
    CELER_EXPECT(state.size() == seeds.size());
    detail::RngSeedExecutor execute_thread{params, state, seeds};
    static KernelLauncher<decltype(execute_thread)> const launch_kernel(
        "rng-reseed");
    launch_kernel(state.size(), stream, execute_thread);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
