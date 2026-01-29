//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/inp/ModelIO.json.cc
//---------------------------------------------------------------------------//
#include "ModelIO.json.hh"

#include "corecel/Assert.hh"
#include "corecel/io/JsonUtils.json.hh"

namespace celeritas
{
namespace inp
{
//---------------------------------------------------------------------------//
//!@{
//! I/O routines for JSON
//! \todo Expand JSON I/O support for all inp::Model fields?

void to_json(nlohmann::json& j, Model const& v)
{
    CELER_VALIDATE(std::holds_alternative<std::string>(v.geometry),
                   << "JSON serialization for model input only supports GDML "
                      "filename");

    j = nlohmann::json{{"geometry", std::get<std::string>(v.geometry)}};
}

void from_json(nlohmann::json const& j, Model& v)
{
    v.geometry = j.at("geometry").get<std::string>();
}

//!@}

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
