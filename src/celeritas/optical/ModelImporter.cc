//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/ModelImporter.cc
//---------------------------------------------------------------------------//
#include "ModelImporter.hh"

#include "corecel/io/EnumStringMapper.hh"
#include "corecel/io/Logger.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/io/ImportOpticalMaterial.hh"
#include "celeritas/mat/MaterialParams.hh"

#include "ImportedMaterials.hh"
#include "ImportedModelAdapter.hh"
#include "MaterialParams.hh"
#include "model/AbsorptionModel.hh"
#include "model/RayleighModel.hh"
#include "model/WavelengthShiftModel.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Construct importer from imported model data and shared material data.
 */
ModelImporter::ModelImporter(ImportData const& data,
                             SPConstMaterial material,
                             SPConstCoreMaterial core_material,
                             UserBuildMap user_build)
    : input_{nullptr, std::move(material), nullptr, std::move(core_material)}
    , user_build_map_(std::move(user_build))
{
    CELER_EXPECT(input_.material);
    CELER_EXPECT(input_.core_material);
    CELER_EXPECT(std::string(data.units) == units::NativeTraits::label());

    input_.imported = ImportedModels::from_import(data);
    input_.import_material = ImportedMaterials::from_import(data);

    CELER_ENSURE(input_.imported);
    CELER_ENSURE(input_.import_material);
}

//---------------------------------------------------------------------------//
/*!
 * Construct without custom user builders.
 */
ModelImporter::ModelImporter(ImportData const& data,
                             SPConstMaterial material,
                             SPConstCoreMaterial core_material)
    : ModelImporter(
          data, std::move(material), std::move(core_material), UserBuildMap{})
{
}

//---------------------------------------------------------------------------//
/*!
 * Create a \c ModelBuilder for the given model class.
 *
 * This may return a null model builder (with a warning) if the user
 * specifically requests that the model be omitted.
 */
auto ModelImporter::operator()(IMC imc) const -> std::optional<ModelBuilder>
{
    // First, look for user-supplied models
    if (auto user_iter = user_build_map_.find(imc);
        user_iter != user_build_map_.end())
    {
        return user_iter->second(input_);
    }

    using BuilderMemFn = ModelBuilder (ModelImporter::*)() const;
    static std::unordered_map<IMC, BuilderMemFn> const builtin_build{
        {IMC::absorption, &ModelImporter::build_absorption},
        {IMC::rayleigh, &ModelImporter::build_rayleigh},
        {IMC::wls, &ModelImporter::build_wls},
        {IMC::wls2, &ModelImporter::build_wls2},
    };

    // Next, try built-in models
    auto iter = builtin_build.find(imc);
    CELER_VALIDATE(iter != builtin_build.end(),
                   << "cannot build unsupported optical model '" << imc << "'");

    BuilderMemFn build_impl{iter->second};
    return (this->*build_impl)();
}

//---------------------------------------------------------------------------//
/*!
 * Create absorption model builder.
 */
auto ModelImporter::build_absorption() const -> ModelBuilder
{
    return AbsorptionModel::make_builder(this->imported());
}

//---------------------------------------------------------------------------//
/*!
 * Create Rayleigh model builder.
 */
auto ModelImporter::build_rayleigh() const -> ModelBuilder
{
    return RayleighModel::make_builder(
        this->imported(),
        RayleighModel::Input{
            this->material(), this->core_material(), this->import_material()});
}

//---------------------------------------------------------------------------//
/*!
 * Create WLS model builder.
 */
auto ModelImporter::build_wls() const -> ModelBuilder
{
    WavelengthShiftModel::Input input;
    input.model = ImportModelClass::wls;
    for (auto mid : range(OptMatId{input_.import_material->num_materials()}))
    {
        input.data.push_back(input_.import_material->wls(mid));
    }
    return WavelengthShiftModel::make_builder(this->imported(),
                                              std::move(input));
}

//---------------------------------------------------------------------------//
/*!
 * Create WLS model builder.
 */
auto ModelImporter::build_wls2() const -> ModelBuilder
{
    WavelengthShiftModel::Input input;
    input.model = ImportModelClass::wls2;
    for (auto mid : range(OptMatId{input_.import_material->num_materials()}))
    {
        input.data.push_back(input_.import_material->wls2(mid));
    }
    return WavelengthShiftModel::make_builder(this->imported(),
                                              std::move(input));
}

//---------------------------------------------------------------------------//
/*!
 * Warn the model is missing and return a null result.
 */
auto WarnAndIgnoreModel::operator()(UserBuildInput const&) const
    -> std::optional<ModelBuilder>
{
    CELER_LOG(warning) << "Omitting '" << model
                       << "' from the optical physics model list";
    return std::nullopt;
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
