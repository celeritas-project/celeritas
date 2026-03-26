//------------------------------ -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/PhysicsParams.cc
//---------------------------------------------------------------------------//
#include "PhysicsParams.hh"

#include "corecel/io/Logger.hh"
#include "corecel/sys/ActionRegistry.hh"

#include "MaterialParams.hh"
#include "MfpBuilder.hh"
#include "Model.hh"
#include "action/DiscreteSelectAction.hh"
#include "model/AbsorptionModel.hh"
#include "model/MieModel.hh"
#include "model/RayleighModel.hh"
#include "model/WavelengthShiftModel.hh"

namespace celeritas
{
namespace optical
{
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
PhysicsParams::PhysicsParams(inp::OpticalBulkPhysics const& input,
                             SPConstMaterials const& materials,
                             SPConstCoreMaterials const& core_materials,
                             SPActionRegistry action_reg)
{
    CELER_EXPECT(materials);
    CELER_EXPECT(core_materials);
    CELER_EXPECT(action_reg);

    CELER_VALIDATE(input, << "no optical bulk models are present");

    // Create and register actions
    {
        // Discrete select action
        discrete_select_
            = std::make_shared<DiscreteSelectAction>(action_reg->next_id());
        action_reg->insert(discrete_select_);

        // Build models
        models_ = this->build_models(
            input, materials, core_materials, *action_reg);
    }

    // Construct data
    HostValue data;
    data.scalars.num_models = models_.size();
    data.scalars.num_materials = materials->num_materials();
    data.scalars.first_model_action = ActionId{1};

    this->build_mfps(*materials, data);

    CELER_ENSURE(data);

    data_ = ParamsDataStore<PhysicsParamsData>{std::move(data)};
}

//---------------------------------------------------------------------------//
/*!
 * Construct optical models and register them in the given registry.
 */
auto PhysicsParams::build_models(inp::OpticalBulkPhysics const& input,
                                 SPConstMaterials materials,
                                 SPConstCoreMaterials core_materials,
                                 ActionRegistry& action_reg) const -> VecModels
{
    VecModels models;
    if (input.absorption)
    {
        models.push_back(std::make_shared<AbsorptionModel>(
            action_reg.next_id(), input.absorption));
        action_reg.insert(models.back());
    }
    if (input.mie)
    {
        models.push_back(std::make_shared<MieModel>(
            action_reg.next_id(), input.mie, materials));
        action_reg.insert(models.back());
    }
    if (input.rayleigh)
    {
        models.push_back(std::make_shared<RayleighModel>(
            action_reg.next_id(), input.rayleigh, materials, core_materials));
        action_reg.insert(models.back());
    }
    if (input.wls)
    {
        models.push_back(std::make_shared<WavelengthShiftModel>(
            action_reg.next_id(), input.wls, materials, "wls"));
        action_reg.insert(models.back());
    }
    if (input.wls2)
    {
        models.push_back(std::make_shared<WavelengthShiftModel>(
            action_reg.next_id(), input.wls2, materials, "wls2"));
        action_reg.insert(models.back());
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
