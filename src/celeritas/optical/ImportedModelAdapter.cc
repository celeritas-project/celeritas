//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/ImportedModelAdapter.cc
//---------------------------------------------------------------------------//
#include "ImportedModelAdapter.hh"

#include "corecel/Macros.hh"
#include "corecel/cont/Range.hh"
#include "corecel/io/Logger.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/io/ImportOpticalMaterial.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Check imported model MFP tables all have the same number of optical
 * materials, and all their model classes are valid.
 */
struct CheckImportModel
{
    std::size_t num_materials;

    bool operator()(ImportOpticalModel const& model) const
    {
        return model.model_class != ImportModelClass::size_
               && model.mfps.size() == num_materials;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Built-in imported model classes.
 *
 * This is a list of imported model classes which can be mapped to built-in
 * models in Celeritas.
 */
auto ImportedModels::builtin_model_classes() -> std::set<IMC> const&
{
    static std::set<IMC> builtins{IMC::absorption, IMC::rayleigh, IMC::wls};
    return builtins;
}

//---------------------------------------------------------------------------//
/*!
 * Construct list of imported model from imported data.
 */
std::shared_ptr<ImportedModels>
ImportedModels::from_import(ImportData const& io)
{
    return std::make_shared<ImportedModels>(io.optical_models);
}

//---------------------------------------------------------------------------//
/*!
 * Construct directly from imported models.
 */
ImportedModels::ImportedModels(std::vector<ImportOpticalModel> models)
    : models_(std::move(models))
{
    CELER_EXPECT(models_.empty()
                 || std::all_of(models_.begin(),
                                models_.end(),
                                CheckImportModel{models_.front().mfps.size()}));

    // Load all built-in IMCs into the map
    for (IMC imc : ImportedModels::builtin_model_classes())
    {
        builtin_id_map_.insert({imc, ImportedModelId{}});
    }

    for (auto model_id : range(models_.size()))
    {
        IMC imc = models_[model_id].model_class;

        // Check if IMC is built-in
        auto iter = builtin_id_map_.find(imc);
        if (iter != builtin_id_map_.end())
        {
            // Update imported model ID
            ImportedModelId& mapped_id = iter->second;
            if (!mapped_id)
            {
                mapped_id = ImportedModelId(model_id);
            }
            else
            {
                CELER_LOG(warning)
                    << "Duplicate built-in optical model '" << to_cstring(imc)
                    << "' data has been imported; will use first imported "
                       "data";
            }
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Get model associated with the given identifier.
 */
ImportOpticalModel const& ImportedModels::model(ImportedModelId mid) const
{
    CELER_EXPECT(mid < models_.size());
    return models_[mid.get()];
}

//---------------------------------------------------------------------------//
/*!
 * Get number of imported models.
 */
auto ImportedModels::num_models() const -> ImportedModelId::size_type
{
    return models_.size();
}

//---------------------------------------------------------------------------//
/*!
 * Get imported model ID for the given built-in model class.
 *
 * Returns an invalid ID if the imported model data is not present.
 */
auto ImportedModels::builtin_model_id(IMC imc) const -> ImportedModelId
{
    CELER_EXPECT(ImportedModels::builtin_model_classes().count(imc) == 1);
    return builtin_id_map_.at(imc);
}

//---------------------------------------------------------------------------//
/*!
 * Create an adapter from imported models for the given model ID.
 */
ImportedModelAdapter::ImportedModelAdapter(ImportedModelId mid,
                                           SPConstImported imported)
    : model_id_(mid), imported_(imported)
{
    CELER_EXPECT(imported_);
    CELER_EXPECT(mid < imported_->num_models());
}

//---------------------------------------------------------------------------//
/*!
 * Get MFP table for the given optical material.
 */
ImportPhysicsVector const& ImportedModelAdapter::mfp(OpticalMaterialId id) const
{
    CELER_EXPECT(id < this->model().mfps.size());
    return this->model().mfps[id.get()];
}

//---------------------------------------------------------------------------//
/*!
 * Get number of optical materials that have MFPs for this model.
 */
OpticalMaterialId::size_type ImportedModelAdapter::num_materials() const
{
    return this->model().mfps.size();
}

//---------------------------------------------------------------------------//
/*!
 * Get model this adapter refers to.
 */
ImportOpticalModel const& ImportedModelAdapter::model() const
{
    return imported_->model(model_id_);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
