//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/TrivialInteractionModel.cu
//---------------------------------------------------------------------------//
#include "TrivialInteractionModel.hh"

#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/action/ActionLauncher.device.hh"
#include "celeritas/optical/action/TrackSlotExecutor.hh"

#include "SurfaceInteractionApplier.hh"
#include "TrivialInteractor.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Launch kernel with device data.
 */
void TrivialInteractionModel::step(CoreParams const& params,
                                   CoreStateDevice& state) const
{
    auto execute = make_surface_physics_executor(
        params.ptr<MemSpace::native>(),
        state.ptr(),
        SurfacePhysicsOrder::interaction,
        this->surface_model_id(),
        SurfaceInteractionApplier{
            TrivialInteractionExecutor{data_.device_ref()}});

    static ActionLauncher<decltype(execute)> const launch_kernel(*this);
    launch_kernel(state, execute);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
