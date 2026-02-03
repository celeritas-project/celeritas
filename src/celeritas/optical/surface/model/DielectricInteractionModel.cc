//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/DielectricInteractionModel.cc
//---------------------------------------------------------------------------//
#include "DielectricInteractionModel.hh"

#include "corecel/data/CollectionBuilder.hh"
#include "celeritas/grid/NonuniformGridBuilder.hh"
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/action/ActionLauncher.hh"
#include "celeritas/optical/action/TrackSlotExecutor.hh"

#include "DielectricInteractor.hh"
#include "SurfaceInteractionApplier.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Construct the model from an ID and a layer map.
 */
DielectricInteractionModel::DielectricInteractionModel(
    SurfaceModelId id, std::map<PhysSurfaceId, InputT> const& layer_map)
    : SurfaceModel(id, "interaction-dielectric")
{
    HostVal<DielectricData> dielectric_data;
    HostVal<UnifiedReflectionData> reflection_data;

    auto interface = CollectionBuilder{&dielectric_data.interface};
    NonuniformGridBuilder build_grid(&reflection_data.reals);

    surfaces_.reserve(layer_map.size());

    for (auto const& [surface, input] : layer_map)
    {
        surfaces_.push_back(surface);

        interface.push_back(input.is_metal ? DielectricInterface::metal
                                           : DielectricInterface::dielectric);

        for (auto mode : range(ReflectionMode::size_))
        {
            CollectionBuilder{&reflection_data.reflection_grids[mode]}.push_back(
                build_grid(input.reflection.reflection_grids[mode]));
        }
    }

    CELER_ENSURE(dielectric_data);
    CELER_ENSURE(reflection_data);

    dielectric_data_
        = ParamsDataStore<DielectricData>{std::move(dielectric_data)};
    reflection_data_
        = ParamsDataStore<UnifiedReflectionData>{std::move(reflection_data)};
}

//---------------------------------------------------------------------------//
/*!
 * Execute model with host data.
 */
void DielectricInteractionModel::step(CoreParams const& params,
                                      CoreStateHost& state) const
{
    launch_action(
        state,
        make_surface_physics_executor(
            params.ptr<MemSpace::native>(),
            state.ptr(),
            SurfacePhysicsOrder::interaction,
            this->surface_model_id(),
            SurfaceInteractionApplier{DielectricInteractor::Executor{
                dielectric_data_.host_ref(), reflection_data_.host_ref()}}));
}

//---------------------------------------------------------------------------//
/*!
 * Execute model with device data.
 */
#if !CELER_USE_DEVICE
void DielectricInteractionModel::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
