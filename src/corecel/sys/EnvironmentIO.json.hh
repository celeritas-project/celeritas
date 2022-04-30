//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file EnvironmentIO.json.hh
//---------------------------------------------------------------------------//
#pragma once

#include <nlohmann/json.hpp>

#include "Environment.hh"
#include "base/Assert.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Read an array from a JSON file.
 */
inline void from_json(const nlohmann::json& j, Environment& value)
{
    CELER_ASSERT(j.is_object());
    for (const auto& el : j.items())
    {
        value.insert({el.key(), el.value().get<std::string>()});
    }
}

//---------------------------------------------------------------------------//
/*!
 * Write an array to a JSON file.
 */
void to_json(nlohmann::json& j, const Environment& value)
{
    j = nlohmann::json::object();
    for (const auto& kvref : value.ordered_environment())
    {
        const Environment::value_type& kv = kvref;
        j[kv.first]                       = kv.second;
    }
}

//---------------------------------------------------------------------------//
} // namespace celeritas
