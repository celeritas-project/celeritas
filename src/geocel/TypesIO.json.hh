//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/TypesIO.json.hh
//---------------------------------------------------------------------------//
#pragma once

#include <nlohmann/json.hpp>

#include "Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
void from_json(nlohmann::json const& j, GeoStatus& value);
void to_json(nlohmann::json& j, GeoStatus const& value);

//---------------------------------------------------------------------------//
}  // namespace celeritas
