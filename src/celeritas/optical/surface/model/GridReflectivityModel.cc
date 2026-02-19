//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/GridReflectivityModel.cc
//---------------------------------------------------------------------------//
#include "GridReflectivityModel.hh"

#include <algorithm>

#include "corecel/data/CollectionBuilder.hh"
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
    using GridId = OpaqueId<NonuniformGridRecord>;

    // Construct surface list
    surfaces_.reserve(layer_map.size());
    std::transform(layer_map.begin(),
                   layer_map.end(),
                   std::back_inserter(surfaces_),
                   [](auto const& layer) { return layer.first; });

    // Build user-defined grids
    HostVal<GridReflectivityData> data;

    // TODO: use the same NonuniformGridBuilder for all collections to dedupe
    // grids

    auto build_reflectivity = NonuniformGridInserter<SubModelId>(
        &data.reals, &data.reflectivity[ReflectivityAction::interact]);
    auto build_transmittance = NonuniformGridInserter<SubModelId>(
        &data.reals, &data.reflectivity[ReflectivityAction::transmit]);

    auto build_efficiency_ids = CollectionBuilder{&data.efficiency_ids};
    auto build_efficiency
        = NonuniformGridInserter<GridId>(&data.reals, &data.efficiency);

    for (auto const& [surface, refl] : layer_map)
    {
        // Build reflectivity grid
        auto const& r_grid = refl.reflectivity;
        CELER_VALIDATE(r_grid,
                       << "a valid reflectivity grid is required for "
                          "user-defined grid reflectivity model");
        CELER_VALIDATE(std::all_of(r_grid.y.begin(),
                                   r_grid.y.end(),
                                   [](double y) { return 0 <= y && y <= 1; }),
                       << "reflectivity grid should all be with unit interval "
                          "[0,1]");
        build_reflectivity(r_grid);

        // Build transmittance grid
        auto const& t_grid = refl.transmittance;
        CELER_VALIDATE(t_grid,
                       << "a valid transmittance grid is required for "
                          "user-defined grid reflectivity model");
        CELER_VALIDATE(std::all_of(t_grid.y.begin(),
                                   t_grid.y.end(),
                                   [](double y) { return 0 <= y && y <= 1; }),
                       << "transmittance grid should all be with unit "
                          "interval [0,1]");
        build_transmittance(t_grid);

        // Optional efficiency grid
        if (auto const& e_grid = refl.efficiency)
        {
            CELER_VALIDATE(
                std::all_of(e_grid.y.begin(),
                            e_grid.y.end(),
                            [](double y) { return 0 <= y && y <= 1; }),
                << "efficiency grid should all be with unit interval "
                   "[0,1]");

            GridId e_grid_id = build_efficiency(e_grid);
            build_efficiency_ids.push_back(e_grid_id);
        }
        else
        {
            build_efficiency_ids.push_back(GridId{});
        }
    }

    CELER_ENSURE(data);
    CELER_ENSURE(data.reflectivity[ReflectivityAction::interact].size()
                 == layer_map.size());
    CELER_ENSURE(data.reflectivity[ReflectivityAction::transmit].size()
                 == layer_map.size());
    CELER_ENSURE(data.efficiency_ids.size() == layer_map.size());

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
