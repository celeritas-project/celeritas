//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/detail/ScintOffloadAction.cu
//---------------------------------------------------------------------------//
#include "ScintOffloadAction.hh"

#include "corecel/Assert.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "celeritas/global/ActionLauncher.device.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"

#include "OpticalGenAlgorithms.hh"
#include "ScintOffloadExecutor.hh"
#include "../ScintillationParams.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Launch a kernel to generate optical distribution data post-step.
 */
void ScintOffloadAction::offload(CoreParams const& core_params,
                                 CoreStateDevice& core_state) const
{
    auto& step_state
        = get<OffloadStepState<MemSpace::native>>(core_state.aux(), step_id_);
    auto& gen_state
        = get<GeneratorState<MemSpace::native>>(core_state.aux(), gen_id_);

    TrackExecutor execute{
        core_params.ptr<MemSpace::native>(),
        core_state.ptr(),
        detail::ScintOffloadExecutor{scintillation_->device_ref(),
                                     gen_state.store.ref(),
                                     step_state.store.ref(),
                                     gen_state.buffer_size}};
    static ActionLauncher<decltype(execute)> const launch_kernel(*this);
    launch_kernel(core_state, execute);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
