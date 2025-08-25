//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/SurfacePhysicsParams.cc
//---------------------------------------------------------------------------//
#include "SurfacePhysicsParams.hh"

#include "corecel/data/CollectionBuilder.hh"
#include "corecel/sys/ActionRegistry.hh"
#include "celeritas/inp/SurfacePhysics.hh"

namespace celeritas
{
namespace optical
{

template<class T>
struct FakeModelBuilder
{
    FakeModelBuilder(std::vector<T> const&) {}

    std::shared_ptr<SurfaceModel>
    operator()(::celeritas::SurfaceModel::SurfaceModelId) const
    {
        return nullptr;
    }
};

template<class T>
void from_import_model(SurfacePhysicsParams::Input& input,
                       SurfacePhysicsStep step,
                       std::map<inp::SurfaceLayer, T> const& model_map)
{
    if (!model_map.empty())
    {
        SurfacePhysicsParams::SurfaceModelId model_id(
            input.model_builders[step].size());
        std::vector<T> parameters;
        for (auto const& [layer, model] : model_map)
        {
            parameters.push_back(model);

            CELER_EXPECT(layer < input.surfaces.size());
            auto& interface_model
                = input.surfaces[layer.get()].interface_models.front()[step];
            CELER_VALIDATE(!interface_model,
                           << " only one surface " << to_cstring(step)
                           << " model valid per surface");
            interface_model = model_id;
        }
        input.model_builders[step].push_back(FakeModelBuilder<T>(parameters));
    }
}

auto SurfacePhysicsParams::Input::from_import(inp::SurfacePhysics const& sp)
    -> Input
{
    CELER_EXPECT(sp);

    Input input;

    input.surfaces.resize(sp.roughness.polished.size()
                              + sp.roughness.smear.size()
                              + sp.roughness.gaussian.size(),
                          SurfaceInput{{},
                                       {
                                           {
                                               SurfaceModelId{},
                                               SurfaceModelId{},
                                               SurfaceModelId{},
                                           },
                                       }});

    {
        // Load roughness models
        auto step = SurfacePhysicsStep::roughness;
        from_import_model(input, step, sp.roughness.polished);
        from_import_model(input, step, sp.roughness.smear);
        from_import_model(input, step, sp.roughness.gaussian);
    }
    {
        // Load reflectivity models
        auto step = SurfacePhysicsStep::reflectivity;
        from_import_model(input, step, sp.reflectivity.grid);
        from_import_model(input, step, sp.reflectivity.fresnel);
    }
    {
        // Load interaction models
        auto step = SurfacePhysicsStep::interaction;
        from_import_model(input, step, sp.interaction.dielectric_dielectric);
        from_import_model(input, step, sp.interaction.dielectric_metal);
    }

    return input;
}

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

        SurfaceModelId model_id{0};
        for (auto const& builder : builders[step])
        {
            models.push_back(builder(model_id++));
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
        auto build_models = make_builder(&data.model_maps[step].surface_models);
        auto build_model_surfaces
            = make_builder(&data.model_maps[step].internal_surface_ids);

        std::vector<size_type> num_model_surfaces(models_[step].size(), 0);

        for (auto const& surface : surfaces)
        {
            for (auto const& interface : surface.interface_models)
            {
                auto model = interface[step];
                CELER_EXPECT(model < num_model_surfaces.size());
                build_models.push_back(model);
                build_model_surfaces.push_back(SurfaceModel::InternalSurfaceId(
                    num_model_surfaces[model.get()]++));
            }
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
