//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantOpticalPhysicsOptionsIO.json.hh
//---------------------------------------------------------------------------//
#pragma once

#include <nlohmann/json.hpp>

namespace celeritas
{
//---------------------------------------------------------------------------//
struct GeantOpticalPhysicsOptions;

//---------------------------------------------------------------------------//
// Read options from JSON
void from_json(nlohmann::json const& j, GeantOpticalPhysicsOptions& opts);

// Write options to JSON
void to_json(nlohmann::json& j, GeantOpticalPhysicsOptions const& opts);

//---------------------------------------------------------------------------//
}  // namespace celeritas
