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

#include "Model.hh"
#include "ModelBuilder.hh"

#include "detail/MfpBuilder.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * TODO: Temporary discrete select action!
 */
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
/*!
 * Construct physics parameters from imported and shared data.
 *
 * The following actions are first constructed:
 *  - "discrete-select": sample models by XS for discrete interaction
 *
 * Optical models provided by the model builders input are then
 * constructed and registered in the action registry. Finally,
 * scalar data, options, and MFP tables are constructed on the
 * physics data storage.
 */
PhysicsParams::PhysicsParams(Input input)
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
        models_ = this->build_models(input.model_builders, action_reg);
    }

    // Construct data
    HostValue host_data;
    host_data.scalars.num_models = this->num_models();
    host_data.scalars.model_to_action = 1;
    this->build_options(input.options, host_data);
    this->build_mfps(host_data);

    // Copy data to device
    data_ = CollectionMirror<PhysicsParamsData>{std::move(host_data)};

    CELER_ENSURE(discrete_action_->action_id()
                 == host_ref().scalars.discrete_action());
}

//---------------------------------------------------------------------------//
/*!
 * Builds optical models from list of model builders.
 *
 * Models are created and registered in the action registry.
 */
auto PhysicsParams::build_models(std::vector<SPConstModelBuilder> const& builders,
                                 ActionRegistry& action_reg) const -> VecModels
{
    VecModels models;
    models.reserve(builders.size());

    for (auto const& build : builders)
    {
        auto action_id = action_reg.next_id();
        auto model = (*build)(action_id);

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
 * Builds optical physics options.
 */
void PhysicsParams::build_options(Options const& /* options */,
                                  HostValue& /* host_data */) const
{
}

//---------------------------------------------------------------------------//
/*!
 * Builds MFP tables for each optical model.
 *
 * Iterates over the constructed optical models and has them build their MFP
 * grids for each optical material. These grids are then mapped to per-model
 * value tables, and the mappings are stored in the optical PhysicsData.
 */
void PhysicsParams::build_mfps(HostValue& data) const
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

        model.build_mfps(detail::MfpBuilder{&inserter, &grid_ids});

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
