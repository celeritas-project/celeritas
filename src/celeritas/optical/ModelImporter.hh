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
class ImportModel;
class ImportedModels;

//---------------------------------------------------------------------------//
/*!
 * Options used for building optical models.
 */
struct ModelImporterOptions
{
};

//---------------------------------------------------------------------------//

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
    //! Construct from imported and shared data
    ModelImporter(SPConstImported models,
                  UserBuildMap user_build,
                  Options options);

    //! Construct from imported and shared data without custom user builders
    ModelImporter(SPConstImported models, Options options);

    //! Create an optical model from the data
    SPModelBuilder operator()(IMC imc) const;

  private:
    UserBuildInput input_;
    UserBuildMap user_build_map_;

    template<class M>
    SPModelBuilder build_builtin(IMC imc) const
    {
        SPModelBuilder builder = nullptr;

        for (auto model_id : range(ModelId{input_.models->num_models()}))
        {
            if (input_.models->model(model_id).model_class == imc)
            {
                builder = std::make_shared<ImportedModelBuilder<M>>(
                    ImportedModelAdapter{input_.models, model_id});
                break;
            }
        }

        CELER_ENSURE(builder);
        return builder;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Warn about a missing optical model and deliberately skip it.
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
