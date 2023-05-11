//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/alongstep/AlongStepNeutralAction.cu
//---------------------------------------------------------------------------//
#include "AlongStepNeutralAction.hh"

#include "corecel/device_runtime_api.h"
#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/KernelParamCalculator.device.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackLauncher.hh"

#include "detail/AlongStepNeutral.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
__global__ void
along_step_neutral_kernel(CRefPtr<CoreParamsData, MemSpace::device> const params,
                          RefPtr<CoreStateData, MemSpace::device> const state,
                          ActionId const along_step_id)
{
    auto launch = make_along_step_track_launcher(
        *params, *state, along_step_id, detail::along_step_neutral);
    launch(KernelParamCalculator::thread_id());
}
//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Launch the along-step action on device.
 */
void AlongStepNeutralAction::execute(CoreParams const& params,
                                     CoreStateDevice& state) const
{
    CELER_LAUNCH_KERNEL(along_step_neutral,
                        celeritas::device().default_block_size(),
                        state.size(),
                        params.ptr<MemSpace::native>(),
                        state.ptr(),
                        this->action_id());
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
