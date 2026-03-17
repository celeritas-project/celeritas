//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/TypesIO.json.cc
//---------------------------------------------------------------------------//
#include "TypesIO.json.hh"

namespace celeritas
{
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
