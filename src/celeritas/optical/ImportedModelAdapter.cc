//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/ImportedModelAdapter.cc
//---------------------------------------------------------------------------//
#include "ImportedModelAdapter.hh"

#include <set>

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
std::shared_ptr<ImportedModels>
ImportedModels::from_import(ImportData const& data)
{
    using IMC = ImportModelClass;

    std::set<IMC> imported_classes;
    std::vector<ImportOpticalModel> models;

    // Copy imported models from data
    for (auto const& model : data.optical_models)
    {
        models.push_back(model);
        imported_classes.insert(model.model_class);
    }

    // Create built-in imported models if not present
    for (auto imc : std::vector<IMC>{IMC::absorption, IMC::rayleigh})
    {
        if (imported_classes.count(imc) == 0)
        {
            models.push_back(ImportOpticalModel{imc, {}});
            imported_classes.insert(imc);
        }
    }

    return std::make_shared<ImportedModels>(std::move(models),
                                            data.optical_materials);
}

ImportedModels::ImportedModels(std::vector<ImportOpticalModel> models,
                               std::vector<ImportOpticalMaterial> materials)
    : models_(std::move(models)), materials_(std::move(materials))
{
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
