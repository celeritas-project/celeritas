//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/TypesIO.json.cc
//---------------------------------------------------------------------------//
#include "TypesIO.json.hh"

#include "corecel/io/StringEnumMapper.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Read a geometry track state from JSON.
 */
void from_json(nlohmann::json const& j, GeoStatus& value)
{
    static auto const from_string
        = StringEnumMapper<GeoStatus>::from_cstring_func(to_cstring,
                                                         "geo status");
    value = from_string(j.get<std::string>());
}

//---------------------------------------------------------------------------//
/*!
 * Write a geometry track state to JSON.
 */
void to_json(nlohmann::json& j, GeoStatus const& value)
{
    j = std::string{to_cstring(value)};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
