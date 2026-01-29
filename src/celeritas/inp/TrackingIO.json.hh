//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/TrackingIO.json.hh
//---------------------------------------------------------------------------//
#pragma once

#include <nlohmann/json.hpp>

#include "Tracking.hh"

namespace celeritas
{
namespace inp
{
//---------------------------------------------------------------------------//

void to_json(nlohmann::json& j, CoreTrackingLimits const&);
void from_json(nlohmann::json const& j, CoreTrackingLimits&);

void to_json(nlohmann::json& j, OpticalTrackingLimits const&);
void from_json(nlohmann::json const& j, OpticalTrackingLimits&);

void to_json(nlohmann::json& j, Tracking const&);
void from_json(nlohmann::json const& j, Tracking&);

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
