//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/TrivialInteractionModel.cc
//---------------------------------------------------------------------------//
#include "TrivialInteractionModel.hh"

#include "corecel/data/CollectionBuilder.hh"
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/action/ActionLauncher.hh"
#include "celeritas/optical/action/TrackSlotExecutor.hh"

#include "SurfaceInteractionApplier.hh"
#include "TrivialInteractor.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Construct the model from an ID and a layer map.
 */
TrivialInteractionModel::TrivialInteractionModel(
    SurfaceModelId id, std::map<PhysSurfaceId, InputT> const& layer_map)
    : SurfaceModel(id, "interaction-trivial")
{
    HostVal<TrivialInteractionData> data;
    auto build_modes = CollectionBuilder{&data.modes};

    surfaces_.reserve(layer_map.size());

    for (auto const& [surface, mode] : layer_map)
    {
        surfaces_.push_back(surface);
        build_modes.push_back(mode);
    }

    CELER_ENSURE(data);

    data_ = ParamsDataStore<TrivialInteractionData>{std::move(data)};
}

//---------------------------------------------------------------------------//
/*!
 * Execute model with host data.
 */
void TrivialInteractionModel::step(CoreParams const& params,
                                   CoreStateHost& state) const
{
    launch_action(state,
                  make_surface_physics_executor(
                      params.ptr<MemSpace::native>(),
                      state.ptr(),
                      SurfacePhysicsOrder::interaction,
                      this->surface_model_id(),
                      SurfaceInteractionApplier{
                          TrivialInteractionExecutor{data_.host_ref()}}));
}

//---------------------------------------------------------------------------//
/*!
 * Execute model with device data.
 */
#if !CELER_USE_DEVICE
void TrivialInteractionModel::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
