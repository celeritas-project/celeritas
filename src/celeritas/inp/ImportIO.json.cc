//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/ImportIO.json.cc
//---------------------------------------------------------------------------//
#include "ImportIO.json.hh"

#include <variant>

#include "corecel/io/JsonUtils.json.hh"

namespace celeritas
{
namespace inp
{
//---------------------------------------------------------------------------//
//!@{
//! I/O routines for JSON

void to_json(nlohmann::json& j, PhysicsFromFile const& v)
{
    j = {
        json_type_pair("file"),
        CELER_JSON_PAIR(v, input),
    };
}

void from_json(nlohmann::json const& j, PhysicsFromFile& v)
{
    CELER_JSON_LOAD_REQUIRED(j, v, input);
}

void to_json(nlohmann::json& j, PhysicsFromGeant const& v)
{
    //! \todo ignoring data_selection
    j = {
        json_type_pair("geant"),
        CELER_JSON_PAIR(v, ignore_processes),
    };
}

void from_json(nlohmann::json const& j, PhysicsFromGeant& v)
{
    //! \todo ignoring data_selection
    CELER_JSON_LOAD_OPTION(j, v, ignore_processes);
}

void to_json(nlohmann::json& j, PhysicsImport const& v)
{
    j = std::visit([](auto const& p) { return nlohmann::json(p); }, v);
}

void from_json(nlohmann::json const& j, PhysicsImport& v)
{
    CELER_JSON_LOAD_VARIANT(j, v, geant, PhysicsFromGeant);
    CELER_JSON_LOAD_VARIANT(j, v, file, PhysicsFromFile);
    CELER_VALIDATE(false, << "invalid PhysicsImport input");
}

//!@}

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
