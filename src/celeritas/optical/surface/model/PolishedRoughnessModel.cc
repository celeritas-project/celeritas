//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/PolishedRoughnessModel.cc
//---------------------------------------------------------------------------//
#include "PolishedRoughnessModel.hh"

#include <algorithm>

#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/action/ActionLauncher.hh"
#include "celeritas/optical/action/TrackSlotExecutor.hh"

#include "PolishedRoughnessExecutor.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Construct the model from an ID and a layer map.
 */
PolishedRoughnessModel::PolishedRoughnessModel(
    SurfaceModelId id, std::map<PhysSurfaceId, InputT> const& layer_map)
    : SurfaceModel(id, "roughness-polished")
{
    surfaces_.reserve(layer_map.size());
    std::transform(layer_map.begin(),
                   layer_map.end(),
                   std::back_inserter(surfaces_),
                   [](auto const& layer) { return layer.first; });
}

//---------------------------------------------------------------------------//
/*!
 * Execute model with host data.
 */
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

//---------------------------------------------------------------------------//
/*!
 * Execute kernel with device data.
 */
#if !CELER_USE_DEVICE
void PolishedRoughnessModel::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
