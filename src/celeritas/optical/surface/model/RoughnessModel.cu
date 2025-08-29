//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/RoughnessModel.cu
//---------------------------------------------------------------------------//
#include "RoughnessModel.hh"

#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/action/ActionLauncher.device.hh"
#include "celeritas/optical/surface/TrackSlotExecutor.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
template<class E>
void RoughnessModel<E>::step(CoreParams const& /* params */,
                             CoreStateDevice& /* state */) const
{
    /*
    auto execute = make_surface_physics_executor(
            params.ptr<MemSpace::native>(),
            state.ptr(),
            SurfacePhysicsOrder::roughness,
            this->surface_model_id(),
            RoughnessApplier{
                controller_.template make_builder<MemSpace::native>()});

    static ActionLauncher<decltype(execute), SurfaceModel> const
    launch_kernel(*this); launch_kernel(state, execute);
    */
}

template class RoughnessModel<SmearRoughnessModelController>;
template class RoughnessModel<GaussianRoughnessModelController>;
template class RoughnessModel<PolishedRoughnessModelController>;

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
