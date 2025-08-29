//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/SmearRoughnessModel.cu
//---------------------------------------------------------------------------//
#include "SmearRoughnessModel.hh"

#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/action/ActionLauncher.device.hh"
#include "celeritas/optical/surface/TrackSlotExecutor.hh"

#include "SmearRoughnessExecutor.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
void SmearRoughnessModel::step(CoreParams const& params,
                               CoreStateDevice& state) const
{
    auto execute = this->make_executor(
        params, state, SmearRoughnessExecutorBuilder{data_.device_ref()});

    static ActionLauncher<decltype(execute), SurfaceModel> const launch_kernel(
        *this);
    launch_kernel(state, execute);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
