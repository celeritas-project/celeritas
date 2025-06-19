//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/SurfacePhysicsParams.cc
//---------------------------------------------------------------------------//
#include "SurfacePhysicsParams.hh"

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
    CELER_EXPECT(!input.model_builders.empty());
    CELER_EXPECT(input.action_registry);

    // Reserve slots in action registry for models

    // Build facet normal actions
    {
        SurfaceModelBuilder::NormalActionBuilder normal_builder{
            input.action_registry};
        for (auto const& model_builder : input.model_builders)
        {
            model_builder.build_facet_normal_actions(normal_builder);
        }
    }
    // Build calculate reflectivity actions
    {
        SurfaceModelBuilder::ReflectivityActionBuilder refl_builder{
            input.action_registry};
        for (auto const& model_builder : input.model_builders)
        {
            model_builder.build_calc_reflectivity_actions(refl_builder);
        }
    }
    // Build interaction actions
    {
        SurfaceModelBuilder::InteractionActionBuilder interaction_builder{
            input.action_registry};
        for (auto const& model_builder : input.model_builders)
        {
            model_builder.build_interation_actions(interaction_builder);
        }
    }
    // Create and register actions
    {
        auto& action_reg = *input.action_registry;

        // Build models
        models_ = this->build_models(input.model_builders, action_reg);
    }

    // Construct data
    HostVal<SurfacePhysicsParamsData> data;

    CELER_ENSURE(data);

    data_ = CollectionMirror<SurfacePhysicsParamsData>{std::move(data)};
}

//---------------------------------------------------------------------------//
/*!
 */
auto SurfacePhysicsParams::build_models(VecModelBuilders const& model_builders,
                                        ActionRegistry& action_reg) const
    -> VecModels
{
    VecModels models;
    models.reserve(model_builders.size());

    for (auto const& builder : model_builders)
    {
        auto action_id = action_reg.next_id();
        SPConstModel model = builder(action_id);

        CELER_ASSERT(model);
        CELER_ASSERT(model->action_id() == action_id);

        action_reg.insert(model);
        models.push_back(std::move(model));
    }

    CELER_ENSURE(models.size() == model_builders.size());
    return models;
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
