//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/PolishedRoughnessModel.cc
//---------------------------------------------------------------------------//
#include "PolishedRoughnessModel.hh"

#include "celeritas/inp/SurfacePhysics.hh"
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/action/ActionLauncher.hh"
#include "celeritas/optical/surface/TrackSlotExecutor.hh"

#include "PolishedRoughnessExecutor.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//

PolishedRoughnessModel::PolishedRoughnessModel(
    SurfaceModelId model,
    std::vector<PhysSurfaceId> surfaces,
    std::vector<InputT> const&)
    : BuiltinRoughnessModel(model, "polished", std::move(surfaces))
{
}

void PolishedRoughnessModel::step(CoreParams const& params,
                                  CoreStateHost& state) const
{
    launch_action(state,
                  this->make_executor(
                      params, state, PolishedRoughnessExecutorBuilder{}));
}

#if !CELER_USE_DEVICE
void PolishedRoughnessModel::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_IMPLEMENTED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
