//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/generated/PreStepAction.cu
//! \note Auto-generated by gen-action.py: DO NOT MODIFY!
//---------------------------------------------------------------------------//
#include "PreStepAction.hh"

#include "corecel/device_runtime_api.h"
#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/sys/KernelParamCalculator.device.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/Stream.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"
#include "../detail/PreStepActionImpl.hh"

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
__launch_bounds__(1024, 7)
#endif
#endif // CELERITAS_LAUNCH_BOUNDS
pre_step_kernel(
    CRefPtr<CoreParamsData, MemSpace::device> const params,
    RefPtr<CoreStateData, MemSpace::device> const state
)
{
    TrackExecutor execute{*params, *state, detail::pre_step_track};
    execute(KernelParamCalculator::thread_id());
}
}  // namespace

void PreStepAction::execute(CoreParams const& params, CoreStateDevice& state) const
{
    CELER_LAUNCH_KERNEL(pre_step,
                        celeritas::device().default_block_size(),
                        state.size(),
                        celeritas::device().stream(state.stream_id()).get(),
                        params.ptr<MemSpace::native>(),
                        state.ptr());
}

}  // namespace generated
}  // namespace celeritas
