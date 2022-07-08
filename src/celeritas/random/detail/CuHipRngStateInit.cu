//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/random/detail/CuHipRngStateInit.cu
//---------------------------------------------------------------------------//
#include "CuHipRngStateInit.hh"

#include "corecel/device_runtime_api.h"
#include "corecel/Assert.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/KernelParamCalculator.device.hh"

#include "../CuHipRngEngine.hh"

namespace celeritas
{
namespace detail
{
namespace
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//
/*!
 * Initialize the RNG states on device from seeds randomly generated on host.
 */
__global__ void
rng_state_init_kernel(::celeritas::DeviceRef<CuHipRngStateData> const state,
                      ::celeritas::DeviceCRef<CuHipRngInitData> const init)
{
    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (tid.get() < state.size())
    {
        CuHipRngEngine rng(state, tid);
        rng = init.seeds[tid];
    }
}

//---------------------------------------------------------------------------//
} // namespace

//---------------------------------------------------------------------------//
// KERNEL INTERFACE
//---------------------------------------------------------------------------//
/*!
 * Initialize the RNG states on device from seeds randomly generated on host.
 */
void rng_state_init(const ::celeritas::DeviceRef<CuHipRngStateData>& rng,
                    const ::celeritas::DeviceCRef<CuHipRngInitData>& seeds)
{
    CELER_EXPECT(rng.size() == seeds.size());
    CELER_LAUNCH_KERNEL(rng_state_init,
                        celeritas::device().default_block_size(),
                        seeds.size(),
                        rng,
                        seeds);
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
