//------------------------------ -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/PhysicsParams.cc
//---------------------------------------------------------------------------//
#include "PhysicsParams.hh"

#include "corecel/sys/ActionRegistry.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/optical/ModelImporter.hh"

#include "MaterialParams.hh"
#include "MfpBuilder.hh"
#include "Model.hh"
#include "action/DiscreteSelectAction.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Construct with imported data.
 */
std::shared_ptr<PhysicsParams>
PhysicsParams::from_import(ImportData const& data,
                           SPConstCoreMaterials core_materials,
                           SPConstMaterials materials,
                           SPActionRegistry action_reg)
{
    Input input;
    input.materials = materials;
    input.action_registry = action_reg.get();
    ModelImporter importer{data, materials, core_materials};
    for (auto const& model : data.optical_models)
    {
        if (auto builder = importer(model.model_class))
        {
            input.model_builders.push_back(*builder);
        }
    }
    return std::make_shared<PhysicsParams>(std::move(input));
}

//---------------------------------------------------------------------------//
/*!
 * Construct from imported and shared data.
 *
 * The following models are first constructed:
 *  - "discrete-select": sample models by XS for discrete interactions
 *
 * Optical models provided by the model builders input are then constructed and
 * registered in the action registry. Finally, scalar data and MFP tables are
 * constructed on the physics storage data.
 */
PhysicsParams::PhysicsParams(Input input)
{
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
    data.scalars.num_materials = input.materials->num_materials();
    data.scalars.first_model_action = ActionId{1};

    this->build_mfps(*input.materials, data);

    CELER_ENSURE(data);

    data_ = ParamsDataStore<PhysicsParamsData>{std::move(data)};
}

//---------------------------------------------------------------------------//
/*!
 * Construct optical models and register them in the given registry.
 */
auto PhysicsParams::build_models(VecModelBuilders const& model_builders,
                                 ActionRegistry& action_reg) const -> VecModels
{
    VecModels models;
    models.reserve(model_builders.size());

    for (auto const& builder : model_builders)
    {
        if (!builder)
        {
            // if model has no data proceed to the next model
            continue;
        }
        auto action_id = action_reg.next_id();
        SPConstModel model = builder(action_id);

        CELER_ASSERT(model);
        CELER_ASSERT(model->action_id() == action_id);

        action_reg.insert(model);
        models.push_back(std::move(model));
    }

    return models;
}

//---------------------------------------------------------------------------//
/*!
 * Build MFP tables for each model in the host data.
 */
void PhysicsParams::build_mfps(MaterialParams const& mats, HostValue& data) const
{
    for (auto const& model : models_)
    {
        // Build all MFP tables for the model
        MfpBuilder builder(&data.reals, &data.grids);
        for (auto opt_mat : range(OptMatId{mats.num_materials()}))
        {
            model->build_mfps(opt_mat, builder);
        }

        // Build the MFP table from the grid IDs
        CELER_ASSERT(builder.grid_ids().size() == mats.num_materials());
    }
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
