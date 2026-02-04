//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/SmearRoughnessModel.cc
//---------------------------------------------------------------------------//
#include "SmearRoughnessModel.hh"

#include <algorithm>

#include "corecel/data/CollectionBuilder.hh"
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/action/ActionLauncher.hh"
#include "celeritas/optical/action/TrackSlotExecutor.hh"

#include "SmearRoughnessExecutor.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Construct the model from an ID and a layer map.
 */
SmearRoughnessModel::SmearRoughnessModel(
    SurfaceModelId id, std::map<PhysSurfaceId, InputT> const& layer_map)
    : SurfaceModel(id, "roughness-smear")
{
    surfaces_.reserve(layer_map.size());
    std::transform(layer_map.begin(),
                   layer_map.end(),
                   std::back_inserter(surfaces_),
                   [](auto const& layer) { return layer.first; });

    HostVal<SmearRoughnessData> data;
    auto build_roughness = CollectionBuilder{&data.roughness};

    for (auto const& [surface, smear] : layer_map)
    {
        CELER_ENSURE(smear);
        build_roughness.push_back(smear.roughness);
    }

    CELER_ENSURE(data);
    CELER_ENSURE(data.roughness.size() == layer_map.size());

    data_ = ParamsDataStore<SmearRoughnessData>{std::move(data)};
}

//---------------------------------------------------------------------------//
/*!
 * Execute model with host data.
 */
void SmearRoughnessModel::step(CoreParams const& params,
                               CoreStateHost& state) const
{
    launch_action(state,
                  make_surface_physics_executor(
                      params.ptr<MemSpace::native>(),
                      state.ptr(),
                      SurfacePhysicsOrder::roughness,
                      this->surface_model_id(),
                      SmearRoughnessExecutor{data_.host_ref()}));
}

//---------------------------------------------------------------------------//
/*!
 * Execute kernel with device data.
 */
#if !CELER_USE_DEVICE
void SmearRoughnessModel::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
