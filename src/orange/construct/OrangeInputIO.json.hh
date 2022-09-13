//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/construct/OrangeInputIO.json.hh
//---------------------------------------------------------------------------//
#pragma once

#include <nlohmann/json.hpp>

#include "OrangeInput.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//

void from_json(const nlohmann::json& j, SurfaceInput& value);
void from_json(const nlohmann::json& j, VolumeInput& value);
void from_json(const nlohmann::json& j, UnitInput& value);
void from_json(const nlohmann::json& j, OrangeInput& value);

//---------------------------------------------------------------------------//
} // namespace celeritas
