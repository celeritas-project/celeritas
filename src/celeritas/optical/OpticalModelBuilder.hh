//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalModelBuilder.hh
//---------------------------------------------------------------------------//
#pragma once

namespace celeritas
{
//---------------------------------------------------------------------------//
class ImportedOpticalModels;
class ImportData;

//---------------------------------------------------------------------------//
/*!
 * Options used for building optical models.
 */
struct OpticalModelBuilderOptions
{
};

//---------------------------------------------------------------------------//
/*!
 * Construct Celeritas optical models from imported data.
 *
 * This is a factory class which constructs optical models from both user
 * models and built-in models, based on the provided user build map. If no
 * map is provided, then built-in optical models are used by default.
 */
class OpticalModelBuilder
{
  public:
    //!@{
    //! \name Type aliases
    using IOMC = ImportOpticalModelClass;
    using Options = OpticalModelBuilderOptions;
    using SPModel = std::shared_ptr<OpticalModel>;
    using SPConstImported = std::shared_ptr<ImportedOpticalMaterials const>;
    using ActionIdIter = RangeIter<ActionId>;
    //!@}

    //! Input argument for user-provided optical model construction
    struct UserBuildInput
    {
        SPConstImported imported;
    }

    //!@{
    //! \name User builder type aliases
    using UserBuildFunction = std::function<SPProcess(ActionIdIter, UserBuildInput const&)>;
    using UserBuildMap = std::unordered_map<IOMC, UserBuildFunction>;
    //!@}

  public:
    //! Get an ordered set of all available optical models
    static std::set<IOMC> get_all_model_classes();

    //! Construct from imported and shared data
    OpticalModelBuilder(ImportData const& data,
                        UserBuildMap user_build,
                        Options options);

    //! Construct from imported and shared data without custom user builders
    OpticalModelBuilder(ImportData const& data,
                        Options options);

    //! Default destructor
    ~OpticalModelBuilder();

    //! Create an optical model from the data
    SPModel operator()(IOMC iomc, ActionIdIter start_id) const;

  private:
    UserBuildInput input_;
    UserBuildMap user_build_map_;

    SPConstImported const imported() const { return input_.imported; }

    SPModel build_absorption() const;
    SPModel build_rayleigh() const;
    SPModel build_wls() const;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
