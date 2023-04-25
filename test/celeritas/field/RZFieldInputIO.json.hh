//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/RZFieldInputIO.json.hh
//---------------------------------------------------------------------------//
#pragma once

#include <nlohmann/json.hpp>

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
struct RZFieldInput;

// Read field from JSON
void from_json(nlohmann::json const& j, RZFieldInput& opts);

// Write field to JSON
void to_json(nlohmann::json& j, RZFieldInput const& opts);

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
