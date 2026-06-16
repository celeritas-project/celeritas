//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mat/MaterialParamsOutput.cc
//---------------------------------------------------------------------------//
#include "MaterialParamsOutput.hh"

#include <utility>
#include <nlohmann/json.hpp>

#include "corecel/Assert.hh"
#include "corecel/io/JsonPimpl.hh"
#include "celeritas/Types.hh"

#include "MaterialParams.hh"  // IWYU pragma: keep

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from shared material data.
 */
MaterialParamsOutput::MaterialParamsOutput(SPConstMaterialParams material)
    : material_(std::move(material))
{
    CELER_EXPECT(material_);
}

//---------------------------------------------------------------------------//
/*!
 * Write output to the given JSON object.
 */
void MaterialParamsOutput::output(JsonPimpl* j) const
{
    using json = nlohmann::json;

    // Output only counts
    auto obj = json::object();
    auto units_unused = json::object();
    obj["num_isotopes"] = material_->num_isotopes();
    obj["num_elements"] = material_->num_elements();
    obj["num_materials"] = material_->num_materials();
    j->obj = std::move(obj);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
