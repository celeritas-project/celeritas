//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/ImportedModelAdapter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>

#include "celeritas/Types.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/io/ImportOpticalModel.hh"

#include "Types.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Collection of all imported data needed to build optical models.
 *
 * Only one imported model data for built-in optical models is expected.
 */
class ImportedModels
{
  public:
    //!@{
    //! \name Type aliases
    using IMC = ImportModelClass;
    using ImportedModelId = OpaqueId<ImportOpticalModel>;
    using ModelIdMap = std::unordered_map<IMC, ImportedModelId>;
    //!@}

  public:
    // Built-in optical model classes
    constexpr static std::array<IMC, 2> builtin_model_classes();

    // Construct imported model data from general imported data
    static std::shared_ptr<ImportedModels> from_import(ImportData const&);

    // Construct imported model data directly
    explicit ImportedModels(std::vector<ImportOpticalModel> models,
                            std::vector<ImportOpticalMaterial> materials);

    // Get imported model data from the given identifier
    ImportOpticalModel const& model(ImportedModelId mid) const;

    // Number of imported models
    ImportedModelId::size_type num_models() const;

    // Get imported material data from the given identifier
    ImportOpticalMaterial const& material(OpticalMaterialId mat_id) const;

    // Number of imported materials
    OpticalMaterialId::size_type num_materials() const;

    // Mapping from built-in model class to imported model ID
    ModelIdMap const& builtin_id_map() const;

  private:
    std::vector<ImportOpticalModel> models_;
    std::vector<ImportOpticalMaterial> materials_;
    ModelIdMap builtin_id_map_;
};

//---------------------------------------------------------------------------//
/*!
 * For a given optical model, this is an adapter to its imported data.
 *
 * MFP tables take preference in the following order:
 *  1. ImportOpticalModel tables
 *  2. ImportOpticalMaterial tables
 *  3. Tables built by model
 *
 * If the MFP table functions return nullptr, then there is not an
 * associated MFP table, and the builder should move to the next option.
 */
class ImportedModelAdapter
{
  public:
    //!@{
    //! \name Type aliases
    using ImportedModelId = ImportedModels::ImportedModelId;
    using SPConstImported = std::shared_ptr<ImportedModels const>;
    //!@}

  public:
    // Construct adapter for given imported model identifier
    ImportedModelAdapter(SPConstImported imported, ImportedModelId mid);

    // Imported optical material data
    ImportOpticalMaterial const& material(OpticalMaterialId mat) const;

    // Number of imported optical materials
    OpticalMaterialId::size_type num_materials() const;

    // Get imported MFP grid, if available
    ImportPhysicsVector const* imported_mfp(OpticalMaterialId mat) const;

    // Get material MFP grid, if available
    ImportPhysicsVector const* material_mfp(OpticalMaterialId mat) const;

    // Get preferred MFP based on available grids
    ImportPhysicsVector const* preferred_mfp(OpticalMaterialId mat) const;

  private:
    // Model this adapter refers to
    ImportOpticalModel const& model() const;

    ImportedModelId model_;
    SPConstImported imported_;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
