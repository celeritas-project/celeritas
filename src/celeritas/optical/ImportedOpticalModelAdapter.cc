//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/ImportedOpticalModelAdapter.cc
//---------------------------------------------------------------------------//
#include "ImportedOpticalModelAdapter.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct ImportOpticalModels from imported optical material data.
 */
std::shared_ptr<ImportOpticalModels>
ImportOpticalModels::from_import(ImportData const& data)
{
    std::vector<ImportOpticalModel> models = {
        ImportOpticalModel(ImportOpticalModelClass::absorption),
        ImportOpticalModel(ImportOpticalModelClass::rayleigh),
        ImportOpticalModel(ImportOpticalModelClass::wavelength_shifting),
    };

    for (auto const& optical_tuple : data.optical)
    {
        ImportOpticalMaterial const& opt_mat = optical_tuple.second;

        for (ImportOpticalModel& model : models)
        {
            if (auto opt_model_mat = opt_mat.model_material(model.model_class))
            {
                model.lambda_table.physics_vectors.push_back(opt_model_mat->mfp);
            }
            else
            {
                model.lambda_table.physics_vectors.push_back(ImportPhysicsVector{});
            }
        }
    }

    return std::make_share<ImportOpticalModels>(std::move(models));
}

//---------------------------------------------------------------------------//
/*!
 * Construct ImportOpticalModels from a list of imported optical models.
 */
ImportOpticalModels::ImportOpticalModels(std::vector<ImportOpticalModel> io)
    : models_(io), ids_()
{
    for (auto id : range(ImportOpticalModelId{this->size()}))
    {
        ImportOpticalModelClass model_class = models_[id.get()].model_class;

        auto insertion = ids_.insert({model_class, id});
        CELER_VALIDATE(insertion.second,
                << "encountered duplicate imported optical models '"
                << to_cstring(model_class) << "' (there may be "
                << "at most one optical model of a given type)"); 
    }

    CELER_ENSURE(models_.size() == ids_.size());
}

//---------------------------------------------------------------------------//
/*!
 * Retrieve the ImportOpticalModelId associted with the given optical model class
 */
auto ImportOpticalModels::find(key_type model) const -> ImportOpticalModelId
{
    auto iter = ids_.find(model);
    if (iter == ids_.end())
        return {};

    return iter->second;
}

//---------------------------------------------------------------------------//
/*!
 * Construct an imported optical model adapter from the imported optical
 * models and a given optical model class.
 */
ImportedOpticalModelAdapter::ImportedOpticalModelAdapter(SPConstImported imported, ImportedOpticalModelClass model_class)
    : imported_(imported), model_class_(model_class)
{}

//---------------------------------------------------------------------------//
/*!
 * Create a step limit builder from imported data.
 */
auto ImportedOpticalModelAdapter::step_limits(OpticalMaterialId opt_mat) const -> StepLimitBuilder
{
    StepLimitBuilder builder;

    ImportPhysicsTable const& lambda = this->get_lambda();
    if (lambda)
    {
        CELER_ASSERT(opt_mat < lambda.physics_vectors.size());
        ImportPhysicsVector const& mfp = lambda.physics_vectors[opt_mat.get()];
        builder = LinearGridBuilder::from_geant(make_span(mfp.x), make_span(mfp.y));
    }

    return std::move(builder);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
