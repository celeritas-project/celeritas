//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detail/CherenkovOffloadAction.cu
//---------------------------------------------------------------------------//
#include "CherenkovOffloadAction.hh"

#include "corecel/Assert.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "celeritas/global/ActionLauncher.device.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"

#include "CherenkovOffloadExecutor.hh"
#include "OffloadParams.hh"
#include "OpticalGenAlgorithms.hh"
#include "../CherenkovParams.hh"
#include "../MaterialParams.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Launch a kernel to generate optical distribution data post-step.
 */
void CherenkovOffloadAction::pre_generate(CoreParams const& core_params,
                                         CoreStateDevice& core_state) const
{
    auto& state = get<OpticalOffloadState<MemSpace::native>>(core_state.aux(),
                                                             data_id_);
    TrackExecutor execute{
        core_params.ptr<MemSpace::native>(),
        core_state.ptr(),
        detail::CherenkovOffloadExecutor{material_->device_ref(),
                                        cherenkov_->device_ref(),
                                        state.store.ref(),
                                        state.buffer_size}};
    static ActionLauncher<decltype(execute)> const launch_kernel(*this);
    launch_kernel(core_state, execute);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
