//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/model/MollerBhabhaModel.cu
//---------------------------------------------------------------------------//
#include "MollerBhabhaModel.hh"

#include "celeritas/em/executor/MollerBhabhaExecutor.hh"
#include "celeritas/global/ActionLauncher.device.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"
#include "celeritas/phys/InteractionApplier.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Interact with device data.
 */
void MollerBhabhaModel::execute(CoreParams const& params,
                                CoreStateDevice& state) const
{
    auto execute = make_action_track_executor(
        params.ptr<MemSpace::native>(),
        state.ptr(),
        this->action_id(),
        InteractionApplier{MollerBhabhaExecutor{this->device_ref()}});
    static ActionLauncher<decltype(execute)> const launch_kernel(params, *this);
    launch_kernel(state, execute);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
