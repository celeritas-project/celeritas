//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/ModelImporter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include "celeritas/io/ImportOpticalModel.hh"

namespace celeritas
{
class MaterialParams;
struct ImportData;
struct ImportOpticalRayleigh;
struct ImportWavelengthShift;

namespace optical
{
struct ModelBuilder;
class MaterialParams;
class ImportedModels;
class ImportedMaterials;
//---------------------------------------------------------------------------//
/*!
 * Construct Celeritas optical model builders from imported data.
 */
class ModelImporter
{
  public:
    //!@{
    //! \name Type aliases
    using IMC = ImportModelClass;
    using SPConstImported = std::shared_ptr<ImportedModels const>;
    using SPConstMaterial = std::shared_ptr<MaterialParams const>;
    using SPConstImportedMaterial = std::shared_ptr<ImportedMaterials const>;
    using SPConstCoreMaterial
        = std::shared_ptr<::celeritas::MaterialParams const>;
    using SPModelBuilder = std::shared_ptr<ModelBuilder>;
    //!@}

    //! Input argument for user-provided process construction
    struct UserBuildInput
    {
        SPConstImported imported;
        SPConstMaterial material;
        SPConstImportedMaterial import_material;
        SPConstCoreMaterial core_material;
    };

    //!@{
    //! \name User builder type aliases
    using UserBuildFunction
        = std::function<SPModelBuilder(UserBuildInput const&)>;
    using UserBuildMap = std::unordered_map<IMC, UserBuildFunction>;
    //!@}

  public:
    // Construct from imported and shared data with user construction
    ModelImporter(ImportData const& data,
                  SPConstMaterial material,
                  SPConstCoreMaterial core_material,
                  UserBuildMap user_build);

    // Construct without custom user builders
    ModelImporter(ImportData const& data,
                  SPConstMaterial material,
                  SPConstCoreMaterial core_material);

    // Create a model builder from the data
    SPModelBuilder operator()(IMC imc) const;

  private:
    UserBuildInput input_;
    UserBuildMap user_build_map_;

    inline SPConstImported const imported() const { return input_.imported; }
    inline SPConstMaterial const material() const { return input_.material; }
    inline SPConstImportedMaterial const import_material() const
    {
        return input_.import_material;
    }
    inline SPConstCoreMaterial const core_material() const
    {
        return input_.core_material;
    }

    SPModelBuilder build_absorption() const;
    SPModelBuilder build_rayleigh() const;
};

//---------------------------------------------------------------------------//
/*!
 * Warn about a missing optical model and deliberately skip it.
 *
 * May be provided as a custom user build function to \c ModelImporter to
 * skip the construction of an optical model builder.
 */
struct WarnAndIgnoreModel
{
    //!@{
    //! \name Type aliases
    using UserBuildInput = ModelImporter::UserBuildInput;
    using SPModelBuilder = typename ModelImporter::SPModelBuilder;
    //!@}

    // Warn about a missing optical model and ignore it
    SPModelBuilder operator()(UserBuildInput const&) const;

    //! Missing optical model to warn about
    ImportModelClass model;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
