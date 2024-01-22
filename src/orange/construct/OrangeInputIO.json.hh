//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/construct/OrangeInputIO.json.hh
//---------------------------------------------------------------------------//
#pragma once

#include <nlohmann/json.hpp>

#include "../OrangeData.hh"
#include "OrangeInput.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//

void from_json(nlohmann::json const& j, VolumeInput& value);
void to_json(nlohmann::json& j, VolumeInput const& value);

void from_json(nlohmann::json const& j, UnitInput& value);
void to_json(nlohmann::json& j, UnitInput const& value);

void from_json(nlohmann::json const& j, RectArrayInput& value);
void to_json(nlohmann::json& j, RectArrayInput const& value);

template<class T>
void from_json(nlohmann::json const& j, Tolerance<>& value);
template<class T>
void to_json(nlohmann::json& j, Tolerance<T> const& value);

void from_json(nlohmann::json const& j, OrangeInput& value);
void to_json(nlohmann::json& j, OrangeInput const& value);

//---------------------------------------------------------------------------//
}  // namespace celeritas
