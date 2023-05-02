//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/generated/DiscreteSelectAction.cu
//! \note Auto-generated by gen-action.py: DO NOT MODIFY!
//---------------------------------------------------------------------------//
#include "DiscreteSelectAction.hh"

#include "corecel/device_runtime_api.h"
#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/sys/KernelParamCalculator.device.hh"
#include "corecel/sys/Device.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackLauncher.hh"
#include "../detail/DiscreteSelectActionImpl.hh"

namespace celeritas
{
namespace generated
{
namespace
{
__global__ void discrete_select_kernel(
    CRefPtr<CoreParamsData, MemSpace::device> const params,
    RefPtr<CoreStateData, MemSpace::device> const state
)
{
    TrackLauncher launch{*params, *state, detail::discrete_select_track};
    launch(KernelParamCalculator::thread_id());
}
}  // namespace

void DiscreteSelectAction::execute(CoreParams const& params, CoreStateDevice& state) const
{
    CELER_LAUNCH_KERNEL(discrete_select,
                        celeritas::device().default_block_size(),
                        state.size(),
                        params.ptr<MemSpace::native>(),
                        state.ptr());
}

}  // namespace generated
}  // namespace celeritas
