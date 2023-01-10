//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/generated/RayleighInteract.cu
//! \note Auto-generated by gen-interactor.py: DO NOT MODIFY!
//---------------------------------------------------------------------------//
#include "RayleighInteract.hh"

#include "corecel/device_runtime_api.h"
#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/sys/KernelParamCalculator.device.hh"
#include "corecel/sys/Device.hh"
#include "celeritas/em/launcher/RayleighLauncher.hh"
#include "celeritas/phys/InteractionLauncher.hh"

using celeritas::MemSpace;

namespace celeritas
{
namespace generated
{
namespace
{
__global__ void
#if CELERITAS_LAUNCH_BOUNDS
#if CELERITAS_USE_CUDA && (__CUDA_ARCH__ == 700) // Tesla V100-SXM2-16GB
__launch_bounds__(1024, 4)
#endif
#if CELERITAS_USE_HIP && defined(__gfx90a__)
__launch_bounds__(1024, 8)
#endif
#endif // CELERITAS_LAUNCH_BOUNDS
rayleigh_interact_kernel(
    celeritas::RayleighDeviceRef const model_data,
    celeritas::CoreRef<MemSpace::device> const core_data)
{
    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (!(tid < core_data.states.size()))
        return;

    auto launch = celeritas::make_interaction_launcher(
        core_data,
        model_data,
        celeritas::rayleigh_interact_track);
    launch(tid);
}
}  // namespace

void rayleigh_interact(
    celeritas::RayleighDeviceRef const& model_data,
    celeritas::CoreRef<MemSpace::device> const& core_data)
{
    CELER_EXPECT(core_data);
    CELER_EXPECT(model_data);

    CELER_LAUNCH_KERNEL(rayleigh_interact,
                        celeritas::device().default_block_size(),
                        core_data.states.size(),
                        model_data, core_data);
}

}  // namespace generated
}  // namespace celeritas
