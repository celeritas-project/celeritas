//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/model/RayleighModel.cu
//---------------------------------------------------------------------------//
#include "RayleighModel.hh"

#include "celeritas/optical/action/ActionLauncher.device.hh"
#include "celeritas/optical/action/TrackSlotExecutor.hh"

#include "RayleighExecutor.hh"
#include "../CoreParams.hh"
#include "../CoreState.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Interact with device data.
 */
void RayleighModel::step(CoreParams const& core_params,
                         CoreStateDevice& core_state) const
{
    auto execute
        = make_action_thread_executor(core_params.ptr<MemSpace::native>(),
                                      core_state.ptr(),
                                      this->action_id(),
                                      RayleighExecutor{});
    static ActionLauncher<decltype(execute)> const launch_kernel(*this);
    launch_kernel(core_state, execute);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
