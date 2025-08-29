//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/SmearRoughnessModel.cc
//---------------------------------------------------------------------------//
#include "SmearRoughnessModel.hh"

#include "corecel/data/CollectionBuilder.hh"
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/action/ActionLauncher.hh"
#include "celeritas/optical/surface/TrackSlotExecutor.hh"

#include "SmearRoughnessExecutor.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 */
SmearRoughnessModel::SmearRoughnessModel(
    SurfaceModelId model,
    std::map<PhysSurfaceId, inp::SmearRoughness> const& layer_map)
    : SurfaceModel(model, "smear")
{
    HostVal<SmearRoughnessData> data;

    auto build_roughness = make_builder(&data.roughness);

    for (auto const& [surface, smear] : layer_map)
    {
        CELER_EXPECT(surface);
        surfaces_.push_back(surface);

        CELER_EXPECT(smear);
        build_roughness.push_back(smear.roughness);
    }

    CELER_ENSURE(data);

    data_ = CollectionMirror<SmearRoughnessData>{std::move(data)};
}

std::vector<PhysSurfaceId> SmearRoughnessModel::get_surfaces() const
{
    return surfaces_;
}

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

void SmearRoughnessModel::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_IMPLEMENTED("CUDA OR HIP");
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
