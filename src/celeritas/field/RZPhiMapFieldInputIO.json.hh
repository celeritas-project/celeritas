//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/RZPhiMapFieldInputIO.json.hh
//---------------------------------------------------------------------------//
#pragma once

#include <nlohmann/json.hpp>

namespace celeritas
{
//---------------------------------------------------------------------------//
struct RZPhiMapFieldInput;

// Read field from JSON
void from_json(nlohmann::json const& j, RZPhiMapFieldInput& opts);

// Write field to JSON
void to_json(nlohmann::json& j, RZPhiMapFieldInput const& opts);

//---------------------------------------------------------------------------//
}  // namespace celeritas
