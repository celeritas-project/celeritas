//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/model/BraggModel.cu
//---------------------------------------------------------------------------//
#include "BraggModel.hh"

#include "celeritas/em/distribution/BraggICRU73QOEnergyDistribution.hh"
#include "celeritas/em/executor/MuHadIonizationExecutor.hh"
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
void BraggModel::step(CoreParams const& params, CoreStateDevice& state) const
{
    auto execute = make_action_track_executor(
        params.ptr<MemSpace::native>(),
        state.ptr(),
        this->action_id(),
        InteractionApplier{
            MuHadIonizationExecutor<BraggICRU73QOEnergyDistribution>{
                this->device_ref()}});
    static ActionLauncher<decltype(execute)> const launch_kernel(*this);
    launch_kernel(*this, params, state, execute);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
