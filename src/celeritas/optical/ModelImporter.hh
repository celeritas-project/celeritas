//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/ModelBuilder.hh
//---------------------------------------------------------------------------//
#pragma once

#include <functional>
#include <memory>
#include <set>
#include <unordered_map>
#include <vector>

#include "corecel/cont/Range.hh"
#include "celeritas/Types.hh"
#include "celeritas/io/ImportOpticalModel.hh"

#include "ModelBuilder.hh"
#include "Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class ImportData;
class ImportOpticalMaterial;

namespace optical
{
//---------------------------------------------------------------------------//
class Model;
class ImportedModels;

//---------------------------------------------------------------------------//
/*!
 * Options used for building optical models.
 */
struct ModelImporterOptions
{
};

//---------------------------------------------------------------------------//
/*!
 * Construct ModelBuilders from imported data.
 *
 * Provides an interface for creating ModelBuilders for built-in models from
 * imported data. Users may provide custom build functions to override the
 * default behavior. Custom user models should be handled elsewhere since
 * imported models do not distinguish by optical ImportModelClass.
 */
class ModelImporter
{
  public:
    //!@{
    //! \name Type aliases
    using IMC = ImportModelClass;
    using SPModelBuilder = std::shared_ptr<ModelBuilder>;
    using Options = ModelImporterOptions;
    using SPConstImported = std::shared_ptr<ImportedModels const>;
    //!@}

    //! Input argument for user-provided optical model construction
    struct UserBuildInput
    {
        SPConstImported models;
    };

    //!@{
    //! \name User builder type aliases
    using UserBuildFunction
        = std::function<SPModelBuilder(UserBuildInput const&)>;
    using UserBuildMap = std::unordered_map<IMC, UserBuildFunction>;
    //!@}

  public:
    // Construct from imported data
    ModelImporter(SPConstImported models,
                  UserBuildMap user_build,
                  Options options);

    // Construct from imported and shared data without custom user builders
    ModelImporter(SPConstImported models, Options options);

    // Create an optical model builder for the given ImportModelClass
    SPModelBuilder operator()(IMC imc) const;

  private:
    UserBuildInput input_;
    UserBuildMap user_build_map_;
};

//---------------------------------------------------------------------------//
/*!
 * Warn about a missing optical model and deliberately skip it.
 *
 * May be provided as a custom user build function to ModelImporter to
 * skip the construction of an optical model builder.
 */
struct WarnAndIgnoreModel
{
    //!@{
    //! \name Type aliases
    using UserBuildInput = ModelImporter::UserBuildInput;
    using SPModelBuilder = ModelImporter::SPModelBuilder;
    //!@}

    //! Warn about a missing optical model and return a null model
    SPModelBuilder operator()(UserBuildInput const&) const;

    //! Optical model class to warn about
    ImportModelClass model;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
