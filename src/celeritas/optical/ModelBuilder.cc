//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/ModelBuilder.cc
//---------------------------------------------------------------------------//
#include "ModelBuilder.hh"

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/optical/ImportMaterialAdapter.hh"
#include "celeritas/optical/model/AbsorptionModel.hh"
#include "celeritas/optical/model/RayleighModel.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
auto ModelBuilder::get_all_model_classes() -> std::set<IMC>
{
    return std::set<IMC>{IMC::absorption, IMC::rayleigh};
}

ModelBuilder::ModelBuilder(ImportData const& data,
                           UserBuildMap user_build,
                           Options /* options */)
    : input_{ImportedMaterials::from_import(data)}
    , user_build_map_(std::move(user_build))
{
    CELER_EXPECT(input_.optical_materials);
}

ModelBuilder::ModelBuilder(ImportData const& data, Options options)
    : ModelBuilder(data, UserBuildMap{}, options)
{
}

auto ModelBuilder::operator()(IMC imc, ActionIdIter& start_id) const -> SPModel
{
    // First, look for user supplied models
    {
        auto user_iter = user_build_map_.find(imc);
        if (user_iter != user_build_map_.end())
        {
            return user_iter->second(start_id, input_);
        }
    }

    // Fallback to built-in models

    using BuildMemFn = SPModel (ModelBuilder::*)(ActionId) const;
    static std::unordered_map<IMC, BuildMemFn> const builtin_build{
        {IMC::absorption, &ModelBuilder::build_absorption},
        {IMC::rayleigh, &ModelBuilder::build_rayleigh}};

    {
        auto iter = builtin_build.find(imc);
        CELER_VALIDATE(iter != builtin_build.end(),
                       << "cannot built unsupported optical model '"
                       << to_cstring(imc) << "'");
        BuildMemFn build_impl{iter->second};
        auto result = (this->*build_impl)(*start_id++);
        CELER_ENSURE(result);
        return result;
    }
}

auto ModelBuilder::build_absorption(ActionId id) const -> SPModel
{
    return std::make_shared<AbsorptionModel>(id, input_.optical_materials);
}

auto ModelBuilder::build_rayleigh(ActionId id) const -> SPModel
{
    return std::make_shared<RayleighModel>(id, input_.optical_materials);
}

auto WarnAndIgnoreModel::operator()(ActionIdIter, UserBuildInput const&) const
    -> SPModel
{
    CELER_LOG(warning) << "Omitting '" << to_cstring(this->model)
                       << "' from the optical physics model list";
    return nullptr;
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
