//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantPhysicsOptionsIO.json.hh
//---------------------------------------------------------------------------//
#pragma once

#include <nlohmann/json.hpp>

namespace celeritas
{
//---------------------------------------------------------------------------//
struct GeantPhysicsOptions;

//---------------------------------------------------------------------------//
// Read options from JSON
void from_json(nlohmann::json const& j, GeantPhysicsOptions& opts);

// Write options to JSON
void to_json(nlohmann::json& j, GeantPhysicsOptions const& opts);

//---------------------------------------------------------------------------//
}  // namespace celeritas
