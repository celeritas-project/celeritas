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

void from_json(nlohmann::json const& j, SurfaceInput& value);
void from_json(nlohmann::json const& j, VolumeInput& value);
void from_json(nlohmann::json const& j, UnitInput& value);
void from_json(nlohmann::json const& j, OrangeInput& value);

//---------------------------------------------------------------------------//
}  // namespace celeritas
