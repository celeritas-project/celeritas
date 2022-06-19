//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/cont/Label.json.hh
//---------------------------------------------------------------------------//
#pragma once

#include <nlohmann/json.hpp>
#include <sstream>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Read an array from a JSON file.
 */
inline void from_json(const nlohmann::json& j, Label& value)
{
    value = Label::from_separator(j.get<std::string>());
}

//---------------------------------------------------------------------------//
/*!
 * Write an array to a JSON file.
 */
inline void to_json(nlohmann::json& j, const Label& value)
{
    j = to_string(value);
}

//---------------------------------------------------------------------------//
} // namespace celeritas
