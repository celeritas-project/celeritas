//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/ModelImporter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <functional>
#include <memory>
#include <optional>

#include "celeritas/io/ImportOpticalMaterial.hh"
#include "celeritas/io/ImportOpticalModel.hh"

#include "Model.hh"
#include "Types.hh"

namespace celeritas
{
class MaterialParams;
struct ImportData;
struct ImportOpticalParameters;
struct ImportOpticalRayleigh;
struct ImportWavelengthShift;
struct ImportMie;

namespace optical
{
class MaterialParams;
class ImportedModels;
class ImportedMaterials;
//---------------------------------------------------------------------------//
/*!
 * Construct Celeritas optical model builders from imported data.
 *
 * \deprecated To be deleted soon
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
    using ModelBuilder = std::function<std::shared_ptr<Model>(ActionId)>;
    //!@}

  public:
    // Construct without custom user builders
    ModelImporter(ImportData const& data,
                  SPConstMaterial material,
                  SPConstCoreMaterial core_material);

    // Create a model builder from the data
    ModelBuilder operator()(IMC imc) const;

  private:
    UserBuildInput input_;
    ImportOpticalParameters const& params_;

    SPConstImported const& imported() const { return input_.imported; }
    SPConstMaterial const& material() const { return input_.material; }
    SPConstImportedMaterial const& import_material() const
    {
        return input_.import_material;
    }
    SPConstCoreMaterial const& core_material() const
    {
        return input_.core_material;
    }

    ModelBuilder build_absorption() const;
    ModelBuilder build_rayleigh() const;
    ModelBuilder build_wls() const;
    ModelBuilder build_wls2() const;
    ModelBuilder build_mie() const;
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
    using ModelBuilder = ModelImporter::ModelBuilder;
    //!@}

    // Warn about a missing optical model and ignore it
    std::optional<ModelBuilder> operator()(UserBuildInput const&) const;

    //! Missing optical model to warn about
    ImportModelClass model;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
