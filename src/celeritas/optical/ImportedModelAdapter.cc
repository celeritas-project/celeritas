//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/ImportedModelAdapter.cc
//---------------------------------------------------------------------------//
#include "ImportedModelAdapter.hh"

#include <algorithm>

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
    // Initialize built-in IMC map to invalid IDs
    std::fill(
        builtin_id_map_.begin(), builtin_id_map_.end(), ImportedModelId{});

    // Load all built-in IMCs into the map
    for (auto model_id : range(models_.size()))
    {
        auto const& model = models_[model_id];

        // Check imported data is consistent
        CELER_VALIDATE(model.model_class != IMC::size_,
                       << "Invalid imported model class for optical model id '"
                       << model_id << "'");

        // Model MFP vectors may be empty, indicating the model should attempt
        // to build from optical material data.

        CELER_VALIDATE(
            model.mfp_table.size() == models_.front().mfp_table.size(),
            << "Imported optical model id '" << model_id << "' ("
            << to_cstring(model.model_class)
            << ") MFP table has differing number of optical "
               "materials than other imported models");

        // Expect a 1-1 mapping for IMC to imported models
        auto& mapped_id = builtin_id_map_[model.model_class];
        CELER_VALIDATE(!mapped_id,
                       << "Duplicate imported data for built-in optical model "
                          "'"
                       << to_cstring(model.model_class)
                       << "' (at most one built-in optical model of a given "
                          "type should be imported)");

        mapped_id = ImportedModelId(model_id);
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
    CELER_EXPECT(imc != IMC::size_);
    return builtin_id_map_[imc];
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
 * Create an adapter from imported models for the given model class.
 */
ImportedModelAdapter::ImportedModelAdapter(ImportModelClass imc,
                                           SPConstImported imported)
    : imported_(imported)
{
    CELER_EXPECT(imported_);
    model_id_ = imported_->builtin_model_id(imc);
    CELER_VALIDATE(model_id_,
                   << "imported data for optical model '" << to_cstring(imc)
                   << "' is missing");
}

//---------------------------------------------------------------------------//
/*!
 * Get MFP table for the given optical material.
 */
inp::Grid const& ImportedModelAdapter::mfp(OpticalMaterialId id) const
{
    CELER_EXPECT(id < this->model().mfp_table.size());
    return this->model().mfp_table[id.get()];
}

//---------------------------------------------------------------------------//
/*!
 * Get number of optical materials that have MFPs for this model.
 */
OpticalMaterialId::size_type ImportedModelAdapter::num_materials() const
{
    return this->model().mfp_table.size();
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
