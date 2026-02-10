//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/EventsIO.json.hh
//---------------------------------------------------------------------------//
#pragma once

#include <nlohmann/json.hpp>

#include "Events.hh"

namespace celeritas
{
namespace inp
{
//---------------------------------------------------------------------------//

void to_json(nlohmann::json& j, EnergyDistribution const&);
void from_json(nlohmann::json const& j, EnergyDistribution&);

void to_json(nlohmann::json& j, ShapeDistribution const&);
void from_json(nlohmann::json const& j, ShapeDistribution&);

void to_json(nlohmann::json& j, AngleDistribution const&);
void from_json(nlohmann::json const& j, AngleDistribution&);

void to_json(nlohmann::json& j, OpticalPrimaryGenerator const&);
void from_json(nlohmann::json const& j, OpticalPrimaryGenerator&);

void to_json(nlohmann::json& j, OpticalEmGenerator const&);
void from_json(nlohmann::json const& j, OpticalEmGenerator&);

void to_json(nlohmann::json& j, OpticalOffloadGenerator const&);
void from_json(nlohmann::json const& j, OpticalOffloadGenerator&);

void to_json(nlohmann::json& j, OpticalDirectGenerator const&);
void from_json(nlohmann::json const& j, OpticalDirectGenerator&);

void to_json(nlohmann::json& j, OpticalGenerator const&);
void from_json(nlohmann::json const& j, OpticalGenerator&);

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
