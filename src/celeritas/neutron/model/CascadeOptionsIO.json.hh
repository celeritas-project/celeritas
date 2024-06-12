//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/neutron/model/CascadeOptionsIO.json.hh
//---------------------------------------------------------------------------//
#pragma once

#include <nlohmann/json.hpp>

namespace celeritas
{
struct CascadeOptions;
//---------------------------------------------------------------------------//

// Read options from JSON
void from_json(nlohmann::json const& j, CascadeOptions& opts);

// Write options to JSON
void to_json(nlohmann::json& j, CascadeOptions const& opts);

//---------------------------------------------------------------------------//
}  // namespace celeritas
