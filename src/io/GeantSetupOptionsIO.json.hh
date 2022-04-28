//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GeantSetupOptionsIO.json.hh
//---------------------------------------------------------------------------//
#pragma once

#include <nlohmann/json.hpp>

#include "GeantSetup.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//

// Read options from JSON
void from_json(const nlohmann::json& j, GeantSetupOptions& opts);

// Write options to JSON
void to_json(nlohmann::json& j, const GeantSetupOptions& opts);

//---------------------------------------------------------------------------//
} // namespace celeritas
