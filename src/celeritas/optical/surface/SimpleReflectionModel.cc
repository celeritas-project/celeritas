//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/SimpleReflectionModel.cc
//---------------------------------------------------------------------------//
#include "SimpleReflectionModel.hh"

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/action/ActionLauncher.hh"
#include "celeritas/optical/action/TrackSlotExecutor.hh"

#include "SurfaceInteractionApplier.hh"
#include "SurfacePhysicsParams.hh"
#include "SimpleReflectionExecutor.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
auto SimpleReflectionModel::make_builder() -> ModelBuilder
{
    return [](ActionId id) {
        return std::make_shared<SimpleReflectionModel>(id);
    };
}

SimpleReflectionModel::SimpleReflectionModel(ActionId id)
    : SurfaceModel(id, "optical-surface-simple", "simple optical reflection surface model")
{
}

void SimpleReflectionModel::step(CoreParams const& params, CoreStateHost& state) const 
{
    launch_action(state, make_action_thread_executor(params.ptr<MemSpace::native>(),
                                                     state.ptr(),
                                                     this->action_id(),
                                                     SurfaceInteractionApplier{SimpleReflectionExecutor{params.surface()->host_ref()}}));

}

#if !CELER_USE_DEVICE
void SimpleReflectionModel::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
