//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/TypesIO.json.hh
//---------------------------------------------------------------------------//
#pragma once

#include <nlohmann/json.hpp>

#include "Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
void from_json(nlohmann::json const& j, TrackOrder& value);
void to_json(nlohmann::json& j, TrackOrder const& value);

void from_json(nlohmann::json const& j, InterpolationType& value);
void to_json(nlohmann::json& j, InterpolationType const& value);

//---------------------------------------------------------------------------//
}  // namespace celeritas
