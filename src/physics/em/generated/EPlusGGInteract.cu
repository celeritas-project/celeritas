//----------------------------------*-cu-*-----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file EPlusGGInteract.cu
//! \note Auto-generated by gen-interactor.py: DO NOT MODIFY!
//---------------------------------------------------------------------------//
#include "base/device_runtime_api.h"
#include "base/Assert.hh"
#include "base/KernelParamCalculator.device.hh"
#include "comm/Device.hh"
#include "../detail/EPlusGGLauncher.hh"

using namespace celeritas::detail;

namespace celeritas
{
namespace generated
{
namespace
{
__global__ void
#if CELERITAS_LAUNCH_BOUNDS
#if CELERITAS_USE_CUDA && (__CUDA_ARCH__ == 700) // Tesla V100-SXM2-16GB
__launch_bounds__(256, 4)
#endif
#if CELERITAS_USE_HIP && defined(__gfx90a__)
__launch_bounds__(256, 32)
#endif
#endif // CELERITAS_LAUNCH_BOUNDS
eplusgg_interact_kernel(
    const detail::EPlusGGDeviceRef eplusgg_data,
    const ModelInteractRef<MemSpace::device> model)
{
    auto tid = KernelParamCalculator::thread_id();
    if (!(tid < model.states.size()))
        return;

    detail::EPlusGGLauncher<MemSpace::device> launch(eplusgg_data, model);
    launch(tid);
}
} // namespace

void eplusgg_interact(
    const detail::EPlusGGDeviceRef& eplusgg_data,
    const ModelInteractRef<MemSpace::device>& model)
{
    CELER_EXPECT(eplusgg_data);
    CELER_EXPECT(model);
    CELER_LAUNCH_KERNEL(eplusgg_interact,
                        celeritas::device().default_block_size(),
                        model.states.size(),
                        eplusgg_data, model);
}

} // namespace generated
} // namespace celeritas
