//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/PrimaryGeneratorOptionsIO.json.hh
//---------------------------------------------------------------------------//
#pragma once

#include <nlohmann/json.hpp>

#include "PrimaryGeneratorOptions.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//

// Read options from JSON
void from_json(const nlohmann::json& j, PrimaryGeneratorOptions& opts);

// Write options to JSON
void to_json(nlohmann::json& j, const PrimaryGeneratorOptions& opts);

//---------------------------------------------------------------------------//
} // namespace celeritas
