//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/GaussianRoughnessModel.cc
//---------------------------------------------------------------------------//
#include "GaussianRoughnessModel.hh"

#include <algorithm>

#include "corecel/data/CollectionBuilder.hh"
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/action/ActionLauncher.hh"
#include "celeritas/optical/action/TrackSlotExecutor.hh"

#include "GaussianRoughnessExecutor.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Construct the model from an ID and a layer map.
 */
GaussianRoughnessModel::GaussianRoughnessModel(
    SurfaceModelId id, std::map<PhysSurfaceId, InputT> const& layer_map)
    : SurfaceModel(id, "roughness-gaussian")
{
    surfaces_.reserve(layer_map.size());
    std::transform(layer_map.begin(),
                   layer_map.end(),
                   std::back_inserter(surfaces_),
                   [](auto const& layer) { return layer.first; });

    HostVal<GaussianRoughnessData> data;
    auto build_sigma_alpha = CollectionBuilder{&data.sigma_alpha};

    for (auto const& [surface, gaussian] : layer_map)
    {
        CELER_ENSURE(gaussian);
        build_sigma_alpha.push_back(gaussian.sigma_alpha);
    }

    CELER_ENSURE(data);
    CELER_ENSURE(data.sigma_alpha.size() == layer_map.size());

    data_ = CollectionMirror<GaussianRoughnessData>{std::move(data)};
}

//---------------------------------------------------------------------------//
/*!
 * Execute model with host data.
 */
void GaussianRoughnessModel::step(CoreParams const& params,
                                  CoreStateHost& state) const
{
    launch_action(state,
                  make_surface_physics_executor(
                      params.ptr<MemSpace::native>(),
                      state.ptr(),
                      SurfacePhysicsOrder::roughness,
                      this->surface_model_id(),
                      GaussianRoughnessExecutor{data_.host_ref()}));
}

//---------------------------------------------------------------------------//
/*!
 * Execute kernel with device data.
 */
#if !CELER_USE_DEVICE
void GaussianRoughnessModel::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
