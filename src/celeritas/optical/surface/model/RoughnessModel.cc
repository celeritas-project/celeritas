//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/RoughnessModel.cc
//---------------------------------------------------------------------------//
#include "RoughnessModel.hh"

#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/action/ActionLauncher.hh"
#include "celeritas/optical/surface/TrackSlotExecutor.hh"

#include "FakeExecutor.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
template<class T>
void RoughnessModel<T>::step(CoreParams const& params,
                             CoreStateHost& state) const
{
    launch_action(state,
                  make_surface_physics_executor(params.ptr<MemSpace::native>(),
                                                state.ptr(),
                                                SurfacePhysicsOrder::roughness,
                                                this->surface_model_id(),
                                                FakeExecutor{}));
    // RoughnessApplier{
    //     controller_.template make_builder<MemSpace::native>()}));
}

#if !CELER_USE_DEVICE
template<class T>
void RoughnessModel<T>::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_IMPLEMENTED("CUDA OR HIP");
}
#endif

template class RoughnessModel<SmearRoughnessModelController>;
template class RoughnessModel<PolishedRoughnessModelController>;
template class RoughnessModel<GaussianRoughnessModelController>;

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
