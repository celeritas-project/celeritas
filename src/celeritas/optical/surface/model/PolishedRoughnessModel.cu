//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/PolishedRoughnessModel.cu
//---------------------------------------------------------------------------//
#include "PolishedRoughnessModel.hh"

#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/action/ActionLauncher.device.hh"
#include "celeritas/optical/surface/TrackSlotExecutor.hh"

#include "PolishedRoughnessExecutor.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
void PolishedRoughnessModel::step(CoreParams const& params,
                                  CoreStateDevice& state) const
{
    auto execute = this->make_executor(
        params, state, PolishedRoughnessExecutorBuilder{});

    static ActionLauncher<decltype(execute), SurfaceModel> const launch_kernel(
        *this);
    launch_kernel(state, execute);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
