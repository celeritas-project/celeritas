//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/MemRegistryIO.json.hh
//---------------------------------------------------------------------------//
#pragma once

#include <nlohmann/json.hpp>

#include "MemRegistry.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//

// Write one memory diagnostic block to json
void to_json(nlohmann::json& j, MemUsageEntry const& md);
// Write device diagnostics to JSON
void to_json(nlohmann::json& j, MemRegistry const& diagnostics);

//---------------------------------------------------------------------------//
}  // namespace celeritas
