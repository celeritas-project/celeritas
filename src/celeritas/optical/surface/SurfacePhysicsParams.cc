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

template<class M>
std::vector<std::shared_ptr<M const>>
build_models(std::vector<typename M::ModelBuilder> const& builders,
             ActionRegistry& action_reg)
{
    using SPConstM = std::shared_ptr<M const>;

    std::vector<SPConstM> models;
    models.reserve(builders.size());

    for (auto const& build : builders)
    {
        auto action_id = action_reg.next_id();
        SPConstM model = build(action_id);

        CELER_ASSERT(model);
        CELER_ASSERT(model->action_id() == action_id);

        action_reg.insert(model);
        models.push_back(std::move(model));
    }

    CELER_ENSURE(models.size() == builders.size());
    return models;
}

//---------------------------------------------------------------------------//
/*!
 */
SurfacePhysicsParams::SurfacePhysicsParams(Input input)
{
    CELER_EXPECT(input.action_registry);

    auto& action_reg = *input.action_registry;

    // Init boundary action
    {
        init_boundary_action_
            = std::make_shared<InitBoundaryAction>(action_reg.next_id());
        CELER_ASSERT(init_boundary_action_);
        action_reg.insert(init_boundary_action_);
    }

    // Boundary action
    {
        boundary_action_
            = std::make_shared<BoundaryAction>(action_reg.next_id());
        CELER_ASSERT(boundary_action_);
        action_reg.insert(boundary_action_);
    }

    // Register roughness actions
    roughness_models_ = build_models<SurfaceRoughnessModel>(
        input.roughness_model_builders, *input.action_registry);

    // Register reflectivity actions
    reflectivity_models_ = build_models<SurfaceReflectivityModel>(
        input.reflectivity_model_builders, *input.action_registry);

    // Register interaction actions
    interaction_models_ = build_models<SurfaceInteractionModel>(
        input.interaction_model_builders, *input.action_registry);

    // Construct data
    HostVal<SurfacePhysicsParamsData> data;

    // Construct scalars

    // Construct surfaces
    auto build_surface = make_builder(&data.surfaces);
    for (auto const& surface : input.surfaces)
    {
        SurfaceRecord record;
        record.roughness_model
            = roughness_models_[surface.roughness_model.get()]->action_id();
        record.reflectivity_model
            = reflectivity_models_[surface.reflectivity_model.get()]->action_id();
        record.interaction_model
            = interaction_models_[surface.interaction_model.get()]->action_id();
        build_surface.push_back(std::move(record));
    }

    // Finalize data

    CELER_ENSURE(data);

    data_ = CollectionMirror<SurfacePhysicsParamsData>{std::move(data)};
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
