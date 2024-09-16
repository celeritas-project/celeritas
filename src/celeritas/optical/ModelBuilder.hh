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
#include "celeritas/optical/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class ImportData;
class ImportOpticalMaterial;

namespace optical
{
//---------------------------------------------------------------------------//
class Model;
class ImportedMaterials;

//---------------------------------------------------------------------------//
/*!
 * Options used for building optical models.
 */
struct ModelBuilderOptions
{
};

//---------------------------------------------------------------------------//
class ModelBuilder
{
  public:
    //!@{
    //! \name Type aliases
    using IMC = ImportModelClass;
    using SPModel = std::shared_ptr<Model>;
    using Options = ModelBuilderOptions;
    using ActionIdIter = RangeIter<ActionId>;
    using SPConstImported = std::shared_ptr<ImportedMaterials const>;
    //!@}

    //! Input argument for user-provided optical model construction
    struct UserBuildInput
    {
        SPConstImported optical_materials;
    };

    //!@{
    //! \name User builder type aliases
    using UserBuildFunction
        = std::function<SPModel(ActionIdIter, UserBuildInput const&)>;
    using UserBuildMap = std::unordered_map<IMC, UserBuildFunction>;
    //!@}

  public:
    //! Get an ordered set of all optical model classes
    static std::set<IMC> get_all_model_classes();

    //! Construct from imported and shared data
    ModelBuilder(ImportData const& data,
                 UserBuildMap user_build,
                 Options options);

    //! Construct from imported and shared data without custom user builders
    ModelBuilder(ImportData const& data, Options options);

    //! Create an optical model from the data
    SPModel operator()(IMC imc, ActionIdIter& start_id) const;

    //! Imported optical materials
    SPConstImported optical_materials() const
    {
        return input_.optical_materials;
    }

  private:
    UserBuildInput input_;
    UserBuildMap user_build_map_;

    //!@{
    //! \name Helper functions to build specific models
    SPModel build_absorption(ActionId) const;
    SPModel build_rayleigh(ActionId) const;
    //!@}
};

//---------------------------------------------------------------------------//
/*!
 * Warn about a missing optical model and deliberately skip it.
 */
struct WarnAndIgnoreModel
{
    //!@{
    //! \name Type aliases
    using UserBuildInput = ModelBuilder::UserBuildInput;
    using ActionIdIter = ModelBuilder::ActionIdIter;
    using SPModel = ModelBuilder::SPModel;
    //!@}

    //! Warn about a missing optical model and return a null model
    SPModel operator()(ActionIdIter, UserBuildInput const&) const;

    //! Optical model class to warn about
    ImportModelClass model;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
