//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/LabelIO.json.hh
//---------------------------------------------------------------------------//
#pragma once

#include <nlohmann/json.hpp>

#include "Label.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Read a label from a JSON file.
 */
inline void from_json(nlohmann::json const& j, Label& value)
{
    value = Label::from_separator(j.get<std::string>());
}

//---------------------------------------------------------------------------//
/*!
 * Write a label to a JSON file.
 */
inline void to_json(nlohmann::json& j, Label const& value)
{
    j = to_string(value);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
