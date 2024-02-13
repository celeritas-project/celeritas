//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/RunInputIO.json.hh
//---------------------------------------------------------------------------//
#pragma once

#include <nlohmann/json.hpp>

#include "RunInput.hh"

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//

void to_json(nlohmann::json& j, RunInput const& value);
void from_json(nlohmann::json const& j, RunInput& value);

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
