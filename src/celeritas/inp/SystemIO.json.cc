//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/SystemIO.json.cc
//---------------------------------------------------------------------------//
#include "SystemIO.json.hh"

#include "corecel/io/JsonUtils.json.hh"

namespace celeritas
{
namespace inp
{
//---------------------------------------------------------------------------//
//!@{
//! I/O routines for JSON

void to_json(nlohmann::json& j, Device const& v)
{
    j = nlohmann::json{
        CELER_JSON_PAIR(v, stack_size),
        CELER_JSON_PAIR(v, heap_size),
    };
}

void from_json(nlohmann::json const& j, Device& v)
{
    CELER_JSON_LOAD_OPTION(j, v, stack_size);
    CELER_JSON_LOAD_OPTION(j, v, heap_size);
}

void to_json(nlohmann::json& j, System const& v)
{
    j = nlohmann::json{
        CELER_JSON_PAIR(v, environment),
        CELER_JSON_PAIR_OPTIONAL(v, device),
    };
}

void from_json(nlohmann::json const& j, System& v)
{
    CELER_JSON_LOAD_OPTION(j, v, environment);
    CELER_JSON_LOAD_OPTIONAL(j, v, device);
}

//!@}

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
