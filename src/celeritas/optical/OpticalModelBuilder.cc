//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalModelBuilder.cc
//---------------------------------------------------------------------------//
#include "OpticalModelBuilder.hh"

namespace celeritas
{
//  //---------------------------------------------------------------------------//
//  /*!
//   * Get an ordered set of all available optical models.
//   */
//  auto OpticalModelBuilder::get_all_model_classes(std::vector<ImportOpticalModel> const& models) -> std::set<IOMC>
//  {
//      std::set<IOMC> result;
//      for (auto const& model : models)
//      {
//          result.insert(model.model_class);
//      }
//      return result;
//  }

//---------------------------------------------------------------------------//
/*!
 * Construct imported optical model data.
 */
OpticalModelBuilder::OpticalModelBuilder(ImportData const& data,
                                         UserBuildMap user_build,
                                         Options options)
    : input_{ImportedOpticalMaterials::from_import(data)}
    , user_build_map_(std::move(user_build))
{}

OpticalModelBuilder::OpticalModelBuilder(ImportData const& data, Options options)
    : OpticalModelBuilder(data, UserBuildMap{}, std::move(options))
{}

OpticalModelBuilder::~OpticalModelBuilder() = default;

auto OpticalModelBuilder::operator()(IOMC iomc, ActionIdIter start_id) -> SPModel
{
    // First, look for user-supplied models
    {
        auto user_iter = user_build_map_.find(iomc);
        if (user_iter != user_build_map_.end())
        {
            return user_iter->second(start_id, input_);
        }
    }

    using BuilderMemFn = SPModel (OpticalModelBuilder::*)(ActionId);
    static std::unordered_map<IOMC, BuilderMemFn> const builtin_build{
        {IOMC::absorption, &ProcessBuilder::build_absorption},
        {IOMC::rayleigh, &ProcessBuilder::build_rayleigh},
        {IOMC::wavelength_shifting, &ProcessBuilder::build_wls}
    };

    // Next, try built-in models
    {
        auto iter = builtin_build.find(iomc);
        CELER_VALIDATE(iter != builtin_build.end(),
                       << "cannot build unsupported EM process '"
                       << to_cstring(iomc) << "'");
        BuilderMemFn build_imp{iter->second};
        auto result = (this->*build_impl)(*start_id++);
        CELER_ENSURE(result);
        return resutl;
    }
}

auto OpticalModelBuilder::build_absorption(ActionId id) const -> SPModel
{
    return std::make_shared<AbsorptionModel>(id, this->imported());
}

auto OpticalModelBuilder::build_rayleigh(ActionId id) const -> SPModel
{
    return std::make_shared<RayleighModel>(id, this->imported());
}

auto OpticalModelBuilder::build_wls(ActionId id) const -> SPModel
{
    return std::make_shared<WavelengthShiftingModel>(id, this->imported());
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
