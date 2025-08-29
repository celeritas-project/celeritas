//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/PolishedRoughnessModel.cc
//---------------------------------------------------------------------------//
#include "PolishedRoughnessModel.hh"

#include "corecel/data/CollectionBuilder.hh"
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
/*!
 *
 */
PolishedRoughnessModel::PolishedRoughnessModel(
    SurfaceModelId model,
    std::map<PhysSurfaceId, inp::NoRoughness> const& layer_map)
    : SurfaceModel(model, "polished")
{
    for (auto const& [surface, polished] : layer_map)
    {
        CELER_EXPECT(surface);
        surfaces_.push_back(surface);
    }
}

std::vector<PhysSurfaceId> PolishedRoughnessModel::get_surfaces() const
{
    return surfaces_;
}

void PolishedRoughnessModel::step(CoreParams const& params,
                                  CoreStateHost& state) const
{
    launch_action(state,
                  make_surface_physics_executor(params.ptr<MemSpace::native>(),
                                                state.ptr(),
                                                SurfacePhysicsOrder::roughness,
                                                this->surface_model_id(),
                                                PolishedRoughnessExecutor{}));
}

void PolishedRoughnessModel::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_IMPLEMENTED("CUDA OR HIP");
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
