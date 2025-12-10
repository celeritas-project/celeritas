//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/grid/GridTypesIO.json.cc
//---------------------------------------------------------------------------//
#include "GridTypesIO.json.hh"

#include "corecel/io/JsonUtils.json.hh"
#include "corecel/io/StringEnumMapper.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Read interpolation method from JSON.
 */
void from_json(nlohmann::json const& j, InterpolationType& value)
{
    static auto const from_string
        = StringEnumMapper<InterpolationType>::from_cstring_func(
            to_cstring, "interpolation");
    value = from_string(j.get<std::string>());
}

//---------------------------------------------------------------------------//
/*!
 * Write interpolation method to JSON.
 */
void to_json(nlohmann::json& j, InterpolationType const& value)
{
    j = std::string{to_cstring(value)};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
