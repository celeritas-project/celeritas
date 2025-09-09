//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/model/EPlusGGModel.cu
//---------------------------------------------------------------------------//
#include "EPlusGGModel.hh"

#include "celeritas/em/executor/EPlusGGExecutor.hh"
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
void EPlusGGModel::step(CoreParams const& params, CoreStateDevice& state) const
{
    auto execute = make_action_track_executor(
        params.ptr<MemSpace::native>(),
        state.ptr(),
        this->action_id(),
        InteractionApplier{EPlusGGExecutor{this->device_ref()}});
    static ActionLauncher<decltype(execute)> const launch_kernel(*this);
    launch_kernel(*this, params, state, execute);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
