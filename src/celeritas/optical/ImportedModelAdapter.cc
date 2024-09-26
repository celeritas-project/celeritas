//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/ImportedModelAdapter.cc
//---------------------------------------------------------------------------//
#include "ImportedModelAdapter.hh"

#include <set>

#include "corecel/cont/Range.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Built-in optical model classes.
 *
 * A list of optical model classes that are considered built-in, and will
 * be appended to imported model list in from_import if not present.
 */
constexpr auto ImportedModels::builtin_model_classes() -> std::array<IMC, 2>
{
    using IMC = ImportModelClass;
    return {IMC::absorption, IMC::rayleigh};
}

//---------------------------------------------------------------------------//
/*!
 * Construct model data from imported model data.
 *
 * Because MFP data for built-in models like absorption may be stored in
 * optical material data, an empty ImportOpticalModel will be added for
 * such models if not already present. If built-in models are already present
 * in ImportData, then those MFP tables should be used instead.
 */
std::shared_ptr<ImportedModels>
ImportedModels::from_import(ImportData const& data)
{
    using IMC = ImportModelClass;

    std::set<IMC> imported_classes;
    std::vector<ImportOpticalModel> models;

    // Copy imported models from data
    for (auto const& model : data.optical_models)
    {
        models.push_back(model);
        imported_classes.insert(model.model_class);
    }

    // Create built-in imported models if not present
    for (auto imc : std::vector<IMC>{IMC::absorption, IMC::rayleigh})
    {
        if (imported_classes.count(imc) == 0)
        {
            models.push_back(ImportOpticalModel{imc, {}});
            imported_classes.insert(imc);
        }
    }

    return std::make_shared<ImportedModels>(std::move(models),
                                            data.optical_materials);
}

//---------------------------------------------------------------------------//
/*!
 * Construct imported model data directly.
 *
 * Does not attempt to construct built-in models if missing.
 */
ImportedModels::ImportedModels(std::vector<ImportOpticalModel> models,
                               std::vector<ImportOpticalMaterial> materials)
    : models_(std::move(models)), materials_(std::move(materials))
{
    constexpr auto builtins = ImportedModels::builtin_model_classes();

    for (auto mid : range(ImportedModelId{models_.size()}))
    {
        auto const& imc = models_[mid.get()].model_class;
        if (std::find(builtins.begin(), builtins.end(), imc) != builtins.end())
        {
            CELER_VALIDATE(builtin_id_map_.find(imc) == builtin_id_map_.end(),
                           << "multiple ImportOpticalModel data present for "
                              "optical model '"
                           << to_cstring(imc) << "'");
            builtin_id_map_.insert({imc, mid});
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Get imported model data from the given identifier.
 */
ImportOpticalModel const& ImportedModels::model(ImportedModelId mid) const
{
    CELER_EXPECT(mid && mid.get() < models_.size());
    return models_[mid.get()];
}

//---------------------------------------------------------------------------//
/*!
 * Number of imported optical models.
 */
auto ImportedModels::num_models() const -> ImportedModelId::size_type
{
    return models_.size();
}

//---------------------------------------------------------------------------//
/*!
 * Get imported material data from the given identifier.
 */
ImportOpticalMaterial const&
ImportedModels::material(OpticalMaterialId mat_id) const
{
    CELER_EXPECT(mat_id && mat_id.get() < materials_.size());
    return materials_[mat_id.get()];
}

//---------------------------------------------------------------------------//
/*!
 * Number of imported materials.
 */
OpticalMaterialId::size_type ImportedModels::num_materials() const
{
    return materials_.size();
}

//---------------------------------------------------------------------------//
/*!
 * Mapping from built-in model class to imported model identifier.
 *
 * If the built-in model is not present, then its model class is not an
 * element of the map.
 */
auto ImportedModels::builtin_id_map() const -> ModelIdMap const&
{
    return builtin_id_map_;
}

//---------------------------------------------------------------------------//
/*!
 * Construct adapter for the given model identifier in the imported model list.
 */
ImportedModelAdapter::ImportedModelAdapter(SPConstImported imported,
                                           ImportedModelId mid)
    : model_(mid), imported_(imported)
{
    CELER_EXPECT(imported_);
    CELER_EXPECT(model_ && model_ < imported_->num_models());
}

//---------------------------------------------------------------------------//
/*!
 * Retrieve imported material data for the given identifier.
 */
ImportOpticalMaterial const&
ImportedModelAdapter::material(OpticalMaterialId mat) const
{
    CELER_EXPECT(mat && mat < imported_->num_materials());
    return imported_->material(mat);
}

//---------------------------------------------------------------------------//
/*!
 * Number of imported optical materials.
 */
OpticalMaterialId::size_type ImportedModelAdapter::num_materials() const
{
    return imported_->num_materials();
}

//---------------------------------------------------------------------------//
/*!
 * Retrieve imported MFP grid, if available.
 *
 * Imported MFP grids allow users to override optical material data for
 * optical models, or provide grids for custom models.
 *
 * Returns the imported MFP grid for the given optical material. If the
 * material ID is not valid in the imported table, or if the grid exists but
 * is invalid, then nullptr is returned and a different MFP grid should be
 * used.
 */
ImportPhysicsVector const*
ImportedModelAdapter::imported_mfp(OpticalMaterialId mat) const
{
    CELER_EXPECT(mat && mat < imported_->num_materials());

    ImportPhysicsVector const* mfp = nullptr;

    auto const& model = this->model();
    if (mat.get() < model.mfps.size() && model.mfps[mat.get()])
    {
        mfp = &model.mfps[mat.get()];
    }

    return mfp;
}

//---------------------------------------------------------------------------//
/*!
 * Retrieve the MFP grid defined by the optical material, if avaialble.
 *
 * Only applicable for built-in optical models that have MFP grids in material
 * data. Specifically:
 *  - Absorption: ImportAbsorptionData::absorption_length
 *  - Rayleigh: ImportRayleighData::mfp
 * Returns nullptr for all other models.
 */
ImportPhysicsVector const*
ImportedModelAdapter::material_mfp(OpticalMaterialId mat) const
{
    switch (this->model().model_class)
    {
        case ImportModelClass::absorption:
            return &this->material(mat).absorption.absorption_length;
        case ImportModelClass::rayleigh:
            return &this->material(mat).rayleigh.mfp;
        default:
            return nullptr;
    }
}

//---------------------------------------------------------------------------//
/*!
 * Retrieve the preferred MFP grid, if available.
 *
 * Returns the MFP grid for the given material identifier by the following
 * preference:
 *  1. Imported model MFP grid
 *  2. Optical material MFP grid
 * If it returns nullptr, then no valid MFP grid is available and the model
 * is expected to construct its own.
 */
ImportPhysicsVector const*
ImportedModelAdapter::preferred_mfp(OpticalMaterialId mat) const
{
    if (auto const* mfp = this->imported_mfp(mat))
    {
        return mfp;
    }
    else if (auto const* mfp = this->material_mfp(mat))
    {
        return mfp;
    }
    else
    {
        return nullptr;
    }
}

//---------------------------------------------------------------------------//
/*!
 * Helper function to access model this adapter corresponds to.
 */
ImportOpticalModel const& ImportedModelAdapter::model() const
{
    return imported_->model(model_);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
