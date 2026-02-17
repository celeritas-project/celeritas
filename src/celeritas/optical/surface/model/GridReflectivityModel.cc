//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/GridReflectivityModel.cc
//---------------------------------------------------------------------------//
#include "GridReflectivityModel.hh"

#include <algorithm>

#include "corecel/io/Logger.hh"
#include "celeritas/grid/NonuniformGridInserter.hh"
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/action/ActionLauncher.hh"
#include "celeritas/optical/action/TrackSlotExecutor.hh"

#include "GridReflectivityExecutor.hh"
#include "ReflectivityApplier.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Construct the model from an ID and a layer map.
 */
GridReflectivityModel::GridReflectivityModel(
    SurfaceModelId id, std::map<PhysSurfaceId, InputT> const& layer_map)
    : SurfaceModel(id, "reflectivity-grid")
{
    surfaces_.reserve(layer_map.size());
    std::transform(layer_map.begin(),
                   layer_map.end(),
                   std::back_inserter(surfaces_),
                   [](auto const& layer) { return layer.first; });

    HostVal<GridReflectivityData> data;
    auto build_reflectivity
        = NonuniformGridInserter<SubModelId>(&data.reals, &data.reflectivity);

    for (auto const& [surface, refl] : layer_map)
    {
        auto const& grid = refl.reflectivity;
        CELER_VALIDATE(std::all_of(grid.y.begin(),
                                   grid.y.end(),
                                   [](double y) { return 0 <= y && y <= 1; }),
                       << "reflectivity grid should all be with unit interval "
                          "[0,1]");
        build_reflectivity(grid);
    }

    CELER_ENSURE(data);
    CELER_ENSURE(data.reflectivity.size() == layer_map.size());

    data_ = ParamsDataStore<GridReflectivityData>{std::move(data)};
}

//---------------------------------------------------------------------------//
/*!
 * Execute model with host data.
 */
void GridReflectivityModel::step(CoreParams const& params,
                                 CoreStateHost& state) const
{
    launch_action(
        state,
        make_surface_physics_executor(
            params.ptr<MemSpace::native>(),
            state.ptr(),
            SurfacePhysicsOrder::reflectivity,
            this->surface_model_id(),
            ReflectivityApplier{GridReflectivityExecutor{data_.host_ref()}}));
}

//---------------------------------------------------------------------------//
/*!
 * Execute the model with device data.
 */
#if !CELER_USE_DEVICE
void GridReflectivityModel::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
