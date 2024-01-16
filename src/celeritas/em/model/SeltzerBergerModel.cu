//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/model/SeltzerBergerModel.cu
//---------------------------------------------------------------------------//
#include "SeltzerBergerModel.hh"

#include "celeritas/em/executor/SeltzerBergerExecutor.hh"
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
void SeltzerBergerModel::execute(CoreParams const& params,
                                 CoreStateDevice& state) const
{
    auto execute = make_action_track_executor(
        params.ptr<MemSpace::native>(),
        state.ptr(),
        this->action_id(),
        InteractionApplier{SeltzerBergerExecutor{this->device_ref()}});
    static ActionLauncher<decltype(execute)> const launch_kernel(*this);
    launch_kernel(params, state, *this, execute);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
