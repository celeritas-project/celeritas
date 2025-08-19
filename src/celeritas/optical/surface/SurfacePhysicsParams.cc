//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/SurfacePhysicsParams.cc
//---------------------------------------------------------------------------//
#include "SurfacePhysicsParams.hh"

#include "corecel/data/CollectionBuilder.hh"
#include "corecel/sys/ActionRegistry.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 */
SurfacePhysicsParams::SurfacePhysicsParams(Input input)
{
    CELER_EXPECT(input.action_reg);

    // Build actions

    auto& action_reg = *input.action_reg;

    // Init boundary action
    {
        init_boundary_action_
            = std::make_shared<InitBoundaryAction>(action_reg.next_id());
        CELER_ASSERT(init_boundary_action_);
        action_reg.insert(init_boundary_action_);
    }

    models_ = this->build_models(input.model_builders);

    // Post boundary action
    {
        post_boundary_action_
            = std::make_shared<PostBoundaryAction>(action_reg.next_id());
        CELER_ASSERT(post_boundary_action_);
        action_reg.insert(post_boundary_action_);
    }

    // Construct data
    HostVal<SurfacePhysicsParamsData> data;

    // Build surfaces
    this->build_surfaces(input.surfaces, data);

    // Finalize data
    CELER_ENSURE(data);

    data_ = CollectionMirror<SurfacePhysicsParamsData>{std::move(data)};
}

//---------------------------------------------------------------------------//
/*!
 * Build sub-step surface physics models.
 */
auto SurfacePhysicsParams::build_models(
    SurfaceStepArray<VecModelBuilders> const& builders) const
    -> SurfaceStepArray<std::vector<SPModel>>
{
    SurfaceStepArray<std::vector<SPModel>> step_models;
    for (auto step : range(SurfacePhysicsStep::size_))
    {
        auto& models = step_models[step];
        models.reserve(builders[step].size());

        ActionId model_action_id{0};
        for (auto const& builder : builders[step])
        {
            models.push_back(builder(model_action_id++));
            CELER_ASSERT(models.back());
        }
    }

    return step_models;
}

//---------------------------------------------------------------------------//
/*!
 * Build surface data form inputs.
 */
void SurfacePhysicsParams::build_surfaces(
    std::vector<SurfaceInput> const& surfaces,
    HostVal<SurfacePhysicsParamsData>& data) const
{
    auto build_surface = make_builder(&data.surfaces);
    auto build_material = make_builder(&data.subsurface_materials);
    auto build_interface = make_builder(&data.subsurface_interfaces);

    PhysicsSurfaceId next_phys_surface{0};
    for (auto const& surface : surfaces)
    {
        std::vector<PhysicsSurfaceId> phys_ids;
        for ([[maybe_unused]] auto const& interface : surface.interface_models)
        {
            phys_ids.push_back(next_phys_surface++);
        }

        SurfaceRecord record;
        record.subsurface_materials
            = ItemMap<SubsurfaceMaterialId, SubsurfaceMaterialRecordId>(
                build_material.insert_back(surface.materials.begin(),
                                           surface.materials.end()));
        record.subsurface_interfaces
            = ItemMap<SubsurfaceInterfaceId, SubsurfaceInterfaceRecordId>(
                build_interface.insert_back(phys_ids.begin(), phys_ids.end()));
        build_surface.push_back(record);
    }

    for (auto step : range(SurfacePhysicsStep::size_))
    {
        auto build_actions = make_builder(&data.model_maps[step].action_ids);
        auto build_model_surfaces
            = make_builder(&data.model_maps[step].model_surface_ids);

        std::vector<size_type> num_model_surfaces(models_[step].size(), 0);

        for (auto const& surface : surfaces)
        {
            for (auto const& interface : surface.interface_models)
            {
                auto model = interface[step];
                CELER_EXPECT(model < num_model_surfaces.size());
                build_actions.push_back(ActionId{model.get()});
                build_model_surfaces.push_back(SurfaceModel::ModelSurfaceId(
                    num_model_surfaces[model.get()]++));
            }
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
