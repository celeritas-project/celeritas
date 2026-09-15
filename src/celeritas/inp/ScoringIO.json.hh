//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/ScoringIO.json.hh
//---------------------------------------------------------------------------//
#pragma once

#include <nlohmann/json.hpp>

#include "Scoring.hh"

namespace celeritas
{
namespace inp
{
//---------------------------------------------------------------------------//

void to_json(nlohmann::json& j, SimpleCalo const&);
void from_json(nlohmann::json const& j, SimpleCalo&);

void to_json(nlohmann::json& j, Scoring const&);
void from_json(nlohmann::json const& j, Scoring&);

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
