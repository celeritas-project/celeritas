//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/EnvironmentIO.json.hh
//---------------------------------------------------------------------------//
#pragma once

#include <nlohmann/json.hpp>

#include "corecel/Assert.hh"

#include "Environment.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Read an array from a JSON file.
 */
inline void from_json(nlohmann::json const& j, Environment& value)
{
    CELER_ASSERT(j.is_object());
    for (auto const& el : j.items())
    {
        value.insert({el.key(), el.value().get<std::string>()});
    }
}

//---------------------------------------------------------------------------//
/*!
 * Write an array to a JSON file.
 */
void to_json(nlohmann::json& j, Environment const& value)
{
    j = nlohmann::json::object();
    for (auto const& kvref : value.ordered_environment())
    {
        Environment::value_type const& kv = kvref;
        j[kv.first] = kv.second;
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
