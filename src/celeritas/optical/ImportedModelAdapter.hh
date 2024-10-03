//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/ImportedModelAdapter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <set>
#include <unordered_map>
#include <vector>

#include "corecel/OpaqueId.hh"
#include "celeritas/Types.hh"
#include "celeritas/io/ImportOpticalModel.hh"

namespace celeritas
{
struct ImportData;

namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * A collection of imported optical models.
 *
 * Constructs a map of built-in optical model classes to their imported model
 * IDs.
 */
class ImportedModels
{
  public:
    //!@{
    //! \name Type aliases
    using IMC = ImportModelClass;
    using ImportedModelId = OpaqueId<ImportOpticalModel>;
    //!@}

  public:
    // Built-in imported model classes
    static std::set<IMC> const& builtin_model_classes();

    // Construct from imported data
    static std::shared_ptr<ImportedModels> from_import(ImportData const&);

    // Construct directly from imported models
    ImportedModels(std::vector<ImportOpticalModel> models);

    // Get model by identifier
    ImportOpticalModel const& model(ImportedModelId mid) const;

    // Get number of imported models
    ImportedModelId::size_type num_models() const;

    // Get imported model ID for the given built-in model class
    ImportedModelId builtin_model_id(IMC imc) const;

  private:
    std::unordered_map<IMC, ImportedModelId> builtin_id_map_;
    std::vector<ImportOpticalModel> models_;
};

//---------------------------------------------------------------------------//
/*!
 * An adapter for imported model data corresponding to a specific identifier.
 */
class ImportedModelAdapter
{
  public:
    //!@{
    //! \name Type aliases
    using ImportedModelId = typename ImportedModels::ImportedModelId;
    using SPConstImported = std::shared_ptr<ImportedModels const>;
    //!@}

  public:
    // Create an adapter for the given model identifier
    ImportedModelAdapter(ImportedModelId id, SPConstImported imported);

    // Get MFP grid for the optical material
    ImportPhysicsVector const& mfp(OpticalMaterialId id) const;

    // Get number of optical materials
    OpticalMaterialId::size_type num_materials() const;

  private:
    // Get imported model referred to by this adapter
    ImportOpticalModel const& model() const;

    ImportedModelId model_id_;
    SPConstImported imported_;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
