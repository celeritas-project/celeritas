//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/RZMapFieldInputIO.json.cc
//---------------------------------------------------------------------------//
#include "RZMapFieldInputIO.json.hh"

#include <initializer_list>
#include <ostream>
#include <string>
#include <vector>

#include "FieldDriverOptionsIO.json.hh"
#include "RZMapFieldInput.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Read field from JSON.
 */
void from_json(nlohmann::json const& j, RZMapFieldInput& inp)
{
#define RZFI_LOAD(NAME) j.at(#NAME).get_to(inp.NAME)
    RZFI_LOAD(num_grid_z);
    RZFI_LOAD(num_grid_r);
    RZFI_LOAD(min_z);
    RZFI_LOAD(min_r);
    RZFI_LOAD(max_z);
    RZFI_LOAD(max_r);
    RZFI_LOAD(field_z);
    RZFI_LOAD(field_r);
    if (j.contains("driver_options"))
    {
        RZFI_LOAD(driver_options);
    }
#undef RZFI_LOAD
}

//---------------------------------------------------------------------------//
/*!
 * Write field to JSON.
 */
void to_json(nlohmann::json& j, RZMapFieldInput const& inp)
{
#define RZFI_KEY_VALUE(NAME) {#NAME, inp.NAME}
    j = {
        RZFI_KEY_VALUE(num_grid_z),
        RZFI_KEY_VALUE(num_grid_r),
        RZFI_KEY_VALUE(min_z),
        RZFI_KEY_VALUE(min_r),
        RZFI_KEY_VALUE(max_z),
        RZFI_KEY_VALUE(max_r),
        RZFI_KEY_VALUE(field_z),
        RZFI_KEY_VALUE(field_r),
        RZFI_KEY_VALUE(driver_options),
    };
#undef RZFI_KEY_VALUE
}

//---------------------------------------------------------------------------//
// Helper to read the field from a file or stream.
std::istream& operator>>(std::istream& is, RZMapFieldInput& inp)
{
    auto j = nlohmann::json::parse(is);
    j.get_to(inp);
    return is;
}

//---------------------------------------------------------------------------//
// Helper to write the field to a file or stream.
std::ostream& operator<<(std::ostream& os, RZMapFieldInput const& inp)
{
    nlohmann::json j = inp;
    os << j.dump(0);
    return os;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
