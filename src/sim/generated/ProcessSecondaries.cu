//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file sim/generated/ProcessSecondaries.cu
//! \note Auto-generated by gen-trackinit.py: DO NOT MODIFY!
//---------------------------------------------------------------------------//
#include "sim/detail/ProcessSecondariesLauncher.hh"

#include "base/device_runtime_api.h"
#include "base/KernelParamCalculator.device.hh"
#include "comm/Device.hh"

namespace celeritas
{
namespace generated
{
namespace
{
__global__ void
#if CELERITAS_LAUNCH_BOUNDS
#if CELERITAS_USE_CUDA && (__CUDA_ARCH__ == 700) // Tesla V100-SXM2-16GB
__launch_bounds__(256, 5)
#endif
#if CELERITAS_USE_HIP && defined(__gfx90a__)
__launch_bounds__(256, 2)
#endif
#endif // CELERITAS_LAUNCH_BOUNDS
process_secondaries_kernel(
    const ParamsDeviceRef params,
    const StateDeviceRef states,
    const TrackInitStateDeviceRef data)
{
    auto tid = KernelParamCalculator::thread_id();
    if (!(tid < states.size()))
        return;

    detail::ProcessSecondariesLauncher<MemSpace::device> launch(params, states, data);
    launch(tid);
}
} // namespace

void process_secondaries(
    const ParamsDeviceRef& params,
    const StateDeviceRef& states,
    const TrackInitStateDeviceRef& data)
{
    CELER_LAUNCH_KERNEL(
        process_secondaries,
        celeritas::device().default_block_size(),
        states.size(),
        params, states, data);
}

} // namespace generated
} // namespace celeritas
