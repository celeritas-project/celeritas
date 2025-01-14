//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/RZMapFieldInputIO.json.hh
//---------------------------------------------------------------------------//
#pragma once

#include <nlohmann/json.hpp>

namespace celeritas
{
//---------------------------------------------------------------------------//
struct RZMapFieldInput;

// Read field from JSON
void from_json(nlohmann::json const& j, RZMapFieldInput& opts);

// Write field to JSON
void to_json(nlohmann::json& j, RZMapFieldInput const& opts);

//---------------------------------------------------------------------------//
}  // namespace celeritas
