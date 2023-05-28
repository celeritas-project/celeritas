//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-loop/RunnerInputIO.json.hh
//---------------------------------------------------------------------------//
#pragma once

#include <nlohmann/json.hpp>

#include "RunnerInput.hh"

namespace demo_loop
{
//---------------------------------------------------------------------------//

void to_json(nlohmann::json& j, RunnerInput const& value);
void from_json(nlohmann::json const& j, RunnerInput& value);

//---------------------------------------------------------------------------//
}  // namespace demo_loop
