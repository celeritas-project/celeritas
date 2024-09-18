//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalPhysicsParams.cc
//---------------------------------------------------------------------------//
#include "OpticalPhysicsParams.hh"

#include "corecel/data/CollectionBuilder.hh"
#include "celeritas/global/ActionInterface.hh"
#include "celeritas/grid/GenericGridBuilder.hh"
#include "celeritas/phys/detail/DiscreteSelectAction.hh"

#include "OpticalMfpBuilder.hh"
#include "OpticalModel.hh"
#include "OpticalModelBuilder.hh"
#include "OpticalPropertyParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct optical physics parameters from a model builder and input.
 */
OpticalPhysicsParams::OpticalPhysicsParams(
        OpticalModelBuilder const& model_builder,
        Input input)
{
    CELER_EXPECT(input.action_registry);

    // Create volumetric discrete action
    discrete_action_ = std::make_shared<detail::DiscreteSelectAction>(input.action_registry->next_id());
    input.action_registry->insert(discrete_action_);
    
    // Create models
    models_ = this->build_models(model_builder, *input.action_registry);

    // Construct data on host
    HostValue host_data;
    this->build_options(input.options, &host_data);
    this->build_mfp(input.properties, &host_data);

    // Copy data to device
    data_ = CollectionMirror<OpticalPhysicsParamsData>{std::move(host_data)};

    CELER_ENSURE(discrete_action_->action_id() == host_ref().scalars.discrete_action());
}

//---------------------------------------------------------------------------//
/*!
 * Build options from provided optical physics options.
 */
void OpticalPhysicsParams::build_options(Options const&, HostValue*) const
{
    // TODO: construct options
}

//---------------------------------------------------------------------------//
/*!
 * Build optical models from imported data.
 */
auto OpticalPhysicsParams::build_models(OpticalModelBuilder const& model_builder,
                                        ActionRegistry& reg) const -> VecModel
{
    VecModel models;

    for (auto iomc : OpticalModelBuilder::get_all_model_classes())
    {
        auto id_iter = OpticalModelBuilder::ActionIdIter{reg.next_id()};
        SPConstModel model = model_builder(iomc, id_iter);
        
        if (model)
        {
            CELER_ASSERT(model->action_id() == *id_iter++);
            reg.insert(model);
            models.push_back(std::move(model));
        }
        else
        {
            // Warn that no model was created
            // This is ok for when a user disables a model
        }
    }

    return models;
}

//---------------------------------------------------------------------------//
/*!
 * Build mean free paths for all optical models.
 */
void OpticalPhysicsParams::build_mfp(SPConstProperties properties,
                                     HostValue* data) const
{
    GenericGridBuilder reals_builder{&data->reals};
    auto grid_builder = CollectionBuilder{&data->grids};
    auto mfp_builder = CollectionBuilder{&data->mat_model_mfp};

    // Loop over all optical materials
    for (auto opt_mat_id : range(OpticalMaterialId{
             properties->host_ref().refractive_index.size()}))
    {
        OpticalModelMfpBuilder model_mfp_builder{&reals_builder, opt_mat_id};

        // Loop over all optical models
        for (auto const& model : models_)
        {
            model->build_mfp(model_mfp_builder);
        }

        // Insert per-model grids
        auto grid_ids = grid_builder.insert_back(
            model_mfp_builder.grids().begin(), model_mfp_builder.grids().end());

        // Insert per-material per-model grids
        mfp_builder.push_back(ItemMap<OpticalModelId, OpticalValueGridId>{grid_ids});
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
