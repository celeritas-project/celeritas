//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/ModelBuilder.cc
//---------------------------------------------------------------------------//
#include "ModelImporter.hh"

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"
#include "celeritas/io/ImportData.hh"

#include "ImportedModelAdapter.hh"
#include "model/AbsorptionModel.hh"
#include "model/RayleighModel.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
auto ModelImporter::get_available_model_classes(SPConstImported models)
    -> std::set<IMC>
{
    std::set<IMC> model_classes;

    for (auto mid : range(ModelId{models->num_models()}))
    {
        model_classes.insert(models->model(mid).model_class);
    }

    return model_classes;
}

ModelImporter::ModelImporter(SPConstImported data,
                             UserBuildMap user_build,
                             Options /* options */)
    : input_{std::move(data)}, user_build_map_(std::move(user_build))
{
    CELER_EXPECT(input_.models);
}

ModelImporter::ModelImporter(SPConstImported data, Options options)
    : ModelImporter(std::move(data), UserBuildMap{}, options)
{
}

auto ModelImporter::operator()(IMC imc) const -> SPModelBuilder
{
    // First, look for user supplied models
    {
        auto user_iter = user_build_map_.find(imc);
        if (user_iter != user_build_map_.end())
        {
            return user_iter->second(input_);
        }
    }

    // Fallback to built-in models

    using BuildMemFn = SPModelBuilder (ModelImporter::*)(IMC) const;
    static std::unordered_map<IMC, BuildMemFn> const builtin_build{
        {IMC::absorption, &ModelImporter::build_builtin<AbsorptionModel>},
        {IMC::rayleigh, &ModelImporter::build_builtin<RayleighModel>}};

    {
        auto iter = builtin_build.find(imc);
        CELER_VALIDATE(iter != builtin_build.end(),
                       << "cannot create builder for unsupported optical "
                          "model '"
                       << to_cstring(imc) << "'");
        BuildMemFn build_impl{iter->second};
        auto result = (this->*build_impl)(imc);
        CELER_ENSURE(result);
        return result;
    }
}

auto WarnAndIgnoreModel::operator()(UserBuildInput const&) const
    -> SPModelBuilder
{
    CELER_LOG(warning) << "Omitting '" << to_cstring(this->model)
                       << "' from the optical physics model list";
    return nullptr;
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
