//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/OrangeInputIO.json.hh
//---------------------------------------------------------------------------//
#pragma once

#include <nlohmann/json.hpp>

#include "OrangeInput.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//

void from_json(nlohmann::json const& j, VolumeInput& value);
void to_json(nlohmann::json& j, VolumeInput const& value);

void from_json(nlohmann::json const& j, BackgroundInput& value);
void to_json(nlohmann::json& j, BackgroundInput const& value);

void from_json(nlohmann::json const& j, UnitInput& value);
void to_json(nlohmann::json& j, UnitInput const& value);

void from_json(nlohmann::json const& j, RectArrayInput& value);
void to_json(nlohmann::json& j, RectArrayInput const& value);

void from_json(nlohmann::json const& j, OrangeInput& value);
void to_json(nlohmann::json& j, OrangeInput const& value);

//---------------------------------------------------------------------------//
}  // namespace celeritas
