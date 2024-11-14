//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/PhysicsParams.cc
//---------------------------------------------------------------------------//
#include "PhysicsParams.hh"

#include <iostream>

#include "corecel/sys/ActionRegistry.hh"

#include "MaterialParams.hh"
#include "MfpBuilder.hh"
#include "Model.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
class DiscreteSelectAction : public ConcreteAction
{
  public:
    DiscreteSelectAction(ActionId id)
        : ConcreteAction(id,
                         "optical-discrete-select",
                         "Optical discrete selection action")
    {
    }
};

PhysicsParams::PhysicsParams(Input input)
{
    CELER_EXPECT(!input.model_builders.empty());
    CELER_EXPECT(input.materials);
    CELER_EXPECT(input.action_registry);

    // Create and register actions
    {
        auto& action_reg = *input.action_registry;

        // Discrete select action
        discrete_select_
            = std::make_shared<DiscreteSelectAction>(action_reg.next_id());
        action_reg.insert(discrete_select_);

        // Build models
        models_ = this->build_models(input.model_builders, action_reg);
    }

    // Construct data
    HostValue data;
    data.scalars.num_models = models_.size();
    data.scalars.model_to_action = 1;

    this->build_mfps(*input.materials, data);

    data_ = CollectionMirror<PhysicsParamsData>{std::move(data)};
}

auto PhysicsParams::build_models(VecModelBuilders const& model_builders,
                                 ActionRegistry& action_reg) const -> VecModels
{
    VecModels models;
    models.reserve(model_builders.size());

    for (auto const& builder : model_builders)
    {
        auto action_id = action_reg.next_id();
        SPConstModel model = (*builder)(action_id);

        CELER_ASSERT(model);
        CELER_ASSERT(model->action_id() == action_id);

        action_reg.insert(model);
        models.push_back(std::move(model));
    }

    CELER_ENSURE(models.size() == model_builders.size());
    return models;
}

void PhysicsParams::build_mfps(MaterialParams const& mats, HostValue& data) const
{
    auto build_table = make_builder(&data.mfp_tables);

    for (auto const& model : models_)
    {
        MfpBuilder builder(&data.reals, &data.grids);
        for (auto opt_mat : range(OpticalMaterialId{mats.num_materials()}))
        {
            model->build_mfps(opt_mat, builder);
        }

        ValueTable table = builder.grid_ids();
        CELER_ASSERT(table.size() == mats.num_materials());

        build_table.push_back(table);
    }

    CELER_EXPECT(data.mfp_tables.size() == models_.size());
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
