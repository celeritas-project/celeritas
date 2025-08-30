//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/GaussianRoughnessModel.cu
//---------------------------------------------------------------------------//
#include "GaussianRoughnessModel.hh"

#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/action/ActionLauncher.device.hh"
#include "celeritas/optical/surface/TrackSlotExecutor.hh"

#include "GaussianRoughnessExecutor.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Launch kernel on device.
 */
void GaussianRoughnessModel::step(CoreParams const& params,
                                  CoreStateDevice& state) const
{
    auto execute = this->make_executor(
        params, state, GaussianRoughnessExecutor{data_.device_ref()});

    static ActionLauncher<decltype(execute), SurfaceModel> const launch_kernel(
        *this);
    launch_kernel(state, execute);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
