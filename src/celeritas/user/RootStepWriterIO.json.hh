//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/RootStepWriterIO.json.hh
//---------------------------------------------------------------------------//
#pragma once

#include <nlohmann/json.hpp>

namespace celeritas
{
//---------------------------------------------------------------------------//
struct SimpleRootFilterInput;

//---------------------------------------------------------------------------//
// Read options from JSON
void from_json(nlohmann::json const& j, SimpleRootFilterInput& opts);

// Write options to JSON
void to_json(nlohmann::json& j, SimpleRootFilterInput const& opts);

//---------------------------------------------------------------------------//
}  // namespace celeritas
