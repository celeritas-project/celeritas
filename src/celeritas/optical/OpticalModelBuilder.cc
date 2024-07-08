//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalModelBuilder.cc
//---------------------------------------------------------------------------//
#include "OpticalModelBuilder.hh"

#include <unordered_map>

#include "corecel/io/EnumStringMapper.hh"
#include "corecel/io/Logger.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/optical/model/AbsorptionModel.hh"
#include "celeritas/optical/model/OpticalRayleighModel.hh"

#include "OpticalModel.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Get an ordered set of all available optical models.
 */
auto OpticalModelBuilder::get_all_model_classes() -> std::set<IOMC>
{
    return std::set<IOMC>{IOMC::absorption, IOMC::rayleigh};
}

//---------------------------------------------------------------------------//
/*!
 * Construct from imported optical model data.
 */
OpticalModelBuilder::OpticalModelBuilder(ImportData const& data,
                                         UserBuildMap user_build,
                                         Options /* options */)
    : input_{&data.opt_materials}, user_build_map_(std::move(user_build))
{
}

//---------------------------------------------------------------------------//
/*!
 * Construct from imported optical model data without custom user builders.
 */
OpticalModelBuilder::OpticalModelBuilder(ImportData const& data, Options options)
    : OpticalModelBuilder(data, UserBuildMap{}, std::move(options))
{}

//---------------------------------------------------------------------------//
/*!
 * Create an optical model from the data.
 *
 * An ActionIdIter is used instead of just an action ID in the case where
 * user models do not produce an action (e.g. when the user decided to
 * not use a model).
 */
auto OpticalModelBuilder::operator()(IOMC iomc, ActionIdIter start_id) const
    -> SPModel
{
    // First, look for user-supplied models
    {
        auto user_iter = user_build_map_.find(iomc);
        if (user_iter != user_build_map_.end())
        {
            return user_iter->second(start_id, input_);
        }
    }

    using BuilderMemFn = SPModel (OpticalModelBuilder::*)(ActionId) const;
    static std::unordered_map<IOMC, BuilderMemFn> const builtin_build{
        {IOMC::absorption, &OpticalModelBuilder::build_absorption},
        {IOMC::rayleigh, &OpticalModelBuilder::build_rayleigh}
        // , {IOMC::wavelength_shifting, &OpticalModelBuilder::build_wls}
    };

    // Next, try built-in models
    {
        auto iter = builtin_build.find(iomc);
        CELER_VALIDATE(iter != builtin_build.end(),
                       << "cannot build unsupported EM process '"
                       << to_cstring(iomc) << "'");
        BuilderMemFn build_impl{iter->second};
        auto result = (this->*build_impl)(*start_id++);
        CELER_ENSURE(result);
        return result;
    }
}

//---------------------------------------------------------------------------//
/*!
 * Helper function to build an optical absorption model.
 */
auto OpticalModelBuilder::build_absorption(ActionId id) const -> SPModel
{
    return std::make_shared<AbsorptionModel>(id, this->optical_materials());
}

//---------------------------------------------------------------------------//
/*!
 * Helper function to build an optical Rayleigh scattering model.
 */
auto OpticalModelBuilder::build_rayleigh(ActionId id) const -> SPModel
{
    return std::make_shared<OpticalRayleighModel>(id,
                                                  this->optical_materials());
}

//---------------------------------------------------------------------------//
/*!
 * Helper function to build an optical wavelength shifting model.
 */
// auto OpticalModelBuilder::build_wls(ActionId id) const -> SPModel
// {
//     return std::make_shared<WavelengthShiftingModel>(id, input_.wls);
// }

//---------------------------------------------------------------------------//
/*!
 * Warn and return a null optical model.
 *
 * Doesn't increment the action ID iterator.
 */
auto WarnAndIgnoreOpticalModel::operator()(ActionIdIter,
                                           UserBuildInput const&) const
    -> SPModel
{
    CELER_LOG(warning) << "Omitting " << to_cstring(this->model)
                       << " from optical physics model list";
    return nullptr;
}

//---------------------------------------------------------------------------//
/*!
 * Get the string form of the ImportOpticalModelClass enumeration.
 */
char const* to_cstring(ImportOpticalModelClass value)
{
    static EnumStringMapper<ImportOpticalModelClass> const to_cstring_impl{
        "", "absorption", "rayleigh", "wavelength_shifting"};
    return to_cstring_impl(value);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
