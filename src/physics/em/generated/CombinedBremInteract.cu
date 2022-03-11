//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file physics/em/generated/CombinedBremInteract.cu
//! \note Auto-generated by gen-interactor.py: DO NOT MODIFY!
//---------------------------------------------------------------------------//
#include "base/device_runtime_api.h"
#include "base/Assert.hh"
#include "base/KernelParamCalculator.device.hh"
#include "comm/Device.hh"
#include "../detail/CombinedBremLauncher.hh"

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
__launch_bounds__(768, 3)
#endif
#if CELERITAS_USE_HIP && defined(__gfx90a__)
__launch_bounds__(1024, 5)
#endif
#endif // CELERITAS_LAUNCH_BOUNDS
combined_brem_interact_kernel(
    const detail::CombinedBremDeviceRef combined_brem_data,
    const ModelInteractRef<MemSpace::device> model)
{
    auto tid = KernelParamCalculator::thread_id();
    if (!(tid < model.states.size()))
        return;

    detail::CombinedBremLauncher<MemSpace::device> launch(combined_brem_data, model);
    launch(tid);
}
} // namespace

void combined_brem_interact(
    const detail::CombinedBremDeviceRef& combined_brem_data,
    const ModelInteractRef<MemSpace::device>& model)
{
    CELER_EXPECT(combined_brem_data);
    CELER_EXPECT(model);
    CELER_LAUNCH_KERNEL(combined_brem_interact,
                        celeritas::device().default_block_size(),
                        model.states.size(),
                        combined_brem_data, model);
}

} // namespace generated
} // namespace celeritas
