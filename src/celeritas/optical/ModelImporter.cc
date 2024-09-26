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
// IMPORTED MODEL HELPERS
//---------------------------------------------------------------------------//
/*!
 * Helper class used to build optical models that only require an action ID
 * and imported data.
 */
template<class M>
class ImportedModelBuilder : public ModelBuilder
{
  public:
    //!@{
    //! \name Type aliases
    using SPModel = std::shared_ptr<Model>;
    //!@}

  public:
    //! Construct builder with data to be provided to the model
    ImportedModelBuilder(ImportedModelAdapter imported) : imported_(imported)
    {
    }

    //! Construct model with given action identifier
    SPModel operator()(ActionId id) const override
    {
        return std::make_shared<M>(id, imported_);
    }

  private:
    ImportedModelAdapter imported_;
};

//---------------------------------------------------------------------------//
/*!
 * Construct built-in ModelBuilders for the given import model class.
 */
template<class M>
std::shared_ptr<ModelBuilder> build_builtin(ImportedModelAdapter imported)
{
    return std::make_shared<ImportedModelBuilder<M>>(std::move(imported));
}
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
/*!
 * Construct from imported data.
 *
 * The UserBuildMap allows the default behavior to be overridden when
 * constructing a given optical model.
 */
ModelImporter::ModelImporter(SPConstImported data,
                             UserBuildMap user_build,
                             Options /* options */)
    : input_{std::move(data)}, user_build_map_(std::move(user_build))
{
    CELER_EXPECT(input_.models);
}

//---------------------------------------------------------------------------//
/*!
 * Construct from imported data without custom user builders.
 */
ModelImporter::ModelImporter(SPConstImported data, Options options)
    : ModelImporter(std::move(data), UserBuildMap{}, options)
{
}

//---------------------------------------------------------------------------//
/*!
 * Create an optical model builder for the given ImportModelClass.
 *
 * First, attempts to use custom user build functions if available. Otherwise,
 * default built-in methods are attempted.
 *
 * Returned values may be null, indicated the model should not be built.
 */
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

    // Find imported model ID for the built-in model
    auto const& builtin_id_map = input_.models->builtin_id_map();
    auto data_iter = builtin_id_map.find(imc);
    CELER_VALIDATE(data_iter != builtin_id_map.end(),
                   << "no imported data for optical model '" << to_cstring(imc)
                   << "'");
    ImportedModelAdapter model_data{input_.models, data_iter->second};

    // Find the construction method for the built-in model
    using BuildFn = SPModelBuilder (*)(ImportedModelAdapter);
    static std::unordered_map<IMC, BuildFn> const builtin_build{
        {IMC::absorption, &build_builtin<AbsorptionModel>},
        {IMC::rayleigh, &build_builtin<RayleighModel>}};

    auto build_iter = builtin_build.find(imc);
    CELER_VALIDATE(build_iter != builtin_build.end(),
                   << "cannot create builder for unsupported optical model '"
                   << to_cstring(imc) << "'");
    BuildFn build_impl{build_iter->second};

    // Create the model builder
    auto result = (*build_impl)(std::move(model_data));
    CELER_ENSURE(result);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Warn that the optical model is missing and return a null result.
 */
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
