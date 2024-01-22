//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
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
rng_state_init_kernel(DeviceCRef<CuHipRngParamsData> const params,
                      DeviceRef<CuHipRngStateData> const state,
                      DeviceCRef<CuHipRngInitData> const seeds)
{
    auto tid = TrackSlotId{
        celeritas::KernelParamCalculator::thread_id().unchecked_get()};
    if (tid.get() < state.size())
    {
        TrackSlotId tsid{tid.unchecked_get()};
        CuHipRngInitializer init;
        init.seed = seeds.seeds[tsid];
        CuHipRngEngine rng(params, state, tsid);
        rng = init;
    }
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
// KERNEL INTERFACE
//---------------------------------------------------------------------------//
/*!
 * Initialize the RNG states on device from seeds randomly generated on host.
 */
void rng_state_init(DeviceCRef<CuHipRngParamsData> const& params,
                    DeviceRef<CuHipRngStateData> const& state,
                    DeviceCRef<CuHipRngInitData> const& seeds)
{
    CELER_EXPECT(state.size() == seeds.size());
    CELER_LAUNCH_KERNEL(rng_state_init, seeds.size(), 0, params, state, seeds);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
