//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/PhysicsParams.cc
//---------------------------------------------------------------------------//
#include "PhysicsParams.hh"

#include "corecel/io/Logger.hh"
#include "corecel/sys/ActionRegistry.hh"
#include "celeritas/optical/Model.hh"
#include "celeritas/optical/detail/MfpBuilder.hh"

namespace celeritas
{
namespace optical
{

class DiscreteSelectAction : public ConcreteAction
{
  public:
    DiscreteSelectAction(ActionId id)
        : ConcreteAction(
            id, "discrete-select", "Optical discrete selection action")
    {
    }
};

//---------------------------------------------------------------------------//
PhysicsParams::PhysicsParams(ModelBuilder builder, Input input)
{
    CELER_EXPECT(input.action_registry);

    // Create actions and register them
    {
        auto& action_reg = *input.action_registry;

        // Discrete selection action
        discrete_action_
            = std::make_shared<DiscreteSelectAction>(action_reg.next_id());
        action_reg.insert(discrete_action_);

        // Create models
        models_ = this->build_models(builder, action_reg);
    }

    // Construct data
    HostValue host_data;
    this->build_options(input.options, host_data);
    host_data.scalars.num_models = this->num_models();
    this->build_mfp(builder.optical_materials(), host_data);

    // Copy data to device
    data_ = CollectionMirror<PhysicsParamsData>{std::move(host_data)};

    CELER_ENSURE(discrete_action_->action_id()
                 == host_ref().scalars.discrete_action());
}

auto PhysicsParams::build_models(ModelBuilder const& build_model,
                                 ActionRegistry& action_reg) const -> VecModels
{
    VecModels models;

    for (auto imc : ModelBuilder::get_all_model_classes())
    {
        auto id_iter = ModelBuilder::ActionIdIter{action_reg.next_id()};
        auto model = build_model(imc, id_iter);

        if (model)
        {
            CELER_ASSERT(model->action_id() == *id_iter++);

            action_reg.insert(model);
            models.push_back(std::move(model));
        }
        else
        {
            // Deliberately ignored model
            CELER_LOG(debug)
                << "Ignored optical model class '" << to_cstring(imc);
        }
    }

    // May want to allow no models built if users don't want to run optical
    // physics?
    CELER_ENSURE(!models.empty());
    return models;
}

void PhysicsParams::build_options(Options const& /* options */,
                                  HostValue& /* host_data */) const
{
}

void PhysicsParams::build_mfp(SPConstMaterials materials, HostValue& data) const
{
    auto build_grid_id = make_builder(&data.grid_ids);
    auto build_table = make_builder(&data.tables);
    auto build_model_tables = make_builder(&data.model_tables);

    for (auto model_id : range(ModelId{models_.size()}))
    {
        auto const& model = *models_[model_id.get()];

        // Build the per material grids
        GenericGridInserter inserter(&data.reals, &data.grids);

        std::vector<ValueGridId> grid_ids;
        grid_ids.reserve(materials->size());

        for (auto opt_mat_id : range(OpticalMaterialId{materials->size()}))
        {
            model.build_mfp(opt_mat_id,
                            detail::MfpBuilder(&inserter, &grid_ids));
        }

        // Build table from material grids
        ValueTable mfp_table;
        mfp_table.grids
            = build_grid_id.insert_back(grid_ids.begin(), grid_ids.end());
        auto mfp_table_id = build_table.push_back(mfp_table);

        // Build model tables
        ModelTables model_tables;
        model_tables.mfp_table = mfp_table_id;
        auto model_tables_id = build_model_tables.push_back(model_tables);

        CELER_EXPECT(model_tables_id.get() == model_id.get());
    }
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
