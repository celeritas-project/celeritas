//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/ImportedModelAdapter.cc
//---------------------------------------------------------------------------//
#include "ImportedModelAdapter.hh"

#include <algorithm>

#include "corecel/cont/Range.hh"
#include "corecel/io/StreamUtils.hh"
#include "celeritas/inp/OpticalPhysics.hh"
#include "celeritas/io/ImportData.hh"

namespace celeritas
{
namespace optical
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Adapter class to construct models from new bulk physics input.
 *
 * This pulls the MFP and model type from each bulk physics model.
 * Model-specific "material" data is managed separately.
 */
struct VecModelBuilder
{
    std::vector<ImportOpticalModel>& models;
    std::size_t num_materials;

    template<class MM, optical::ImportModelClass IMC>
    void operator()(inp::OpticalBulkModel<MM, IMC> const& model)
    {
        if (!model)
        {
            return;
        }

        ImportOpticalModel iom;
        iom.model_class = model.model_class;
        iom.mfp_table.resize(this->num_materials);
        for (auto&& [opt_mat, model_mat] : model.materials)
        {
            CELER_ASSERT(opt_mat < iom.mfp_table.size());
            iom.mfp_table[opt_mat.get()] = model_mat.mfp;
        }

        this->models.emplace_back(std::move(iom));
    }
};

}  // namespace
//---------------------------------------------------------------------------//
/*!
 * Construct list of imported model from imported data.
 */
std::shared_ptr<ImportedModels>
ImportedModels::from_import(ImportData const& io)
{
    std::vector<ImportOpticalModel> models;
    VecModelBuilder add_model{models, io.optical_materials.size()};
    add_model(io.optical_physics.bulk.absorption);
    add_model(io.optical_physics.bulk.rayleigh);
    add_model(io.optical_physics.bulk.mie);
    add_model(io.optical_physics.bulk.wls);
    add_model(io.optical_physics.bulk.wls2);

    return std::make_shared<ImportedModels>(std::move(models));
}

//---------------------------------------------------------------------------//
/*!
 * Construct directly from imported models.
 */
ImportedModels::ImportedModels(std::vector<ImportOpticalModel> models)
    : models_(std::move(models))
{
    // Initialize built-in IMC map to null IDs
    std::fill(
        builtin_id_map_.begin(), builtin_id_map_.end(), ImportedModelId{});

    // Load all built-in IMCs into the map
    for (auto model_id : range(models_.size()))
    {
        auto const& model = models_[model_id];

        // Check imported data is consistent
        CELER_VALIDATE(model.model_class != IMC::size_,
                       << "invalid imported model class for optical model id '"
                       << model_id << "'");

        // Model MFP vectors may be empty (*may* indicate )
        CELER_VALIDATE(
            model.mfp_table.size() == models_.front().mfp_table.size(),
            << "imported optical model id '" << model_id << "' ("
            << model.model_class
            << ") MFP table has differing number of optical "
               "materials than other imported models");

        // Expect a 1-1 mapping for IMC to imported models
        auto& mapped_id = builtin_id_map_[model.model_class];
        CELER_VALIDATE(!mapped_id,
                       << "duplicate imported data for built-in optical model "
                          "'"
                       << model.model_class
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
 * Returns a null ID if the imported model data is not present.
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
    CELER_EXPECT(imc != ImportModelClass::size_);
    CELER_EXPECT(imported_);
    model_id_ = imported_->builtin_model_id(imc);
    CELER_VALIDATE(model_id_,
                   << "imported data for optical model '" << imc
                   << "' is missing");
}

//---------------------------------------------------------------------------//
/*!
 * Get MFP table for the given optical material.
 */
inp::Grid const& ImportedModelAdapter::mfp(OptMatId id) const
{
    CELER_EXPECT(id < this->model().mfp_table.size());
    return this->model().mfp_table[id.get()];
}

//---------------------------------------------------------------------------//
/*!
 * Get number of optical materials that have MFPs for this model.
 */
OptMatId::size_type ImportedModelAdapter::num_materials() const
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
