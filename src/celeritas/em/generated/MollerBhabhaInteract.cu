//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/generated/MollerBhabhaInteract.cu
//! \note Auto-generated by gen-interactor.py: DO NOT MODIFY!
//---------------------------------------------------------------------------//
#include "MollerBhabhaInteract.hh"

#include "corecel/device_runtime_api.h"
#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/sys/KernelParamCalculator.device.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/Stream.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/em/executor/MollerBhabhaExecutor.hh"
#include "celeritas/phys/InteractionExecutor.hh"

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
moller_bhabha_interact_kernel(
    celeritas::CRefPtr<celeritas::CoreParamsData, MemSpace::device> const params,
    celeritas::RefPtr<celeritas::CoreStateData, MemSpace::device> const state,
    celeritas::MollerBhabhaDeviceRef const model_data,
    celeritas::size_type size)
{
    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (!(tid < size))
        return;

    auto execute = celeritas::make_interaction_executor(
        params, state, celeritas::moller_bhabha_interact_track, model_data);
    execute(tid);
}
}  // namespace

void moller_bhabha_interact(
    celeritas::CoreParams const& params,
    celeritas::CoreState<MemSpace::device>& state,
    celeritas::MollerBhabhaDeviceRef const& model_data)
{
    CELER_EXPECT(model_data);

    CELER_LAUNCH_KERNEL(moller_bhabha_interact,
                        celeritas::device().default_block_size(),
                        state.size(),
                        celeritas::device().stream(state.stream_id()).get(),
                        params.ptr<MemSpace::native>(),
                        state.ptr(),
                        model_data,
                        state.size());
}

}  // namespace generated
}  // namespace celeritas
