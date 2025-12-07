//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/inp/IO.json.hh
//---------------------------------------------------------------------------//
#pragma once

#include <nlohmann/json.hpp>

#include "Import.hh"

namespace celeritas
{
namespace inp
{
//---------------------------------------------------------------------------//

void to_json(nlohmann::json& j, InlineSingletons const&);
void from_json(nlohmann::json const& j, InlineSingletons&);

void to_json(nlohmann::json& j, OrangeGeoFromGeant const&);
void from_json(nlohmann::json const& j, OrangeGeoFromGeant&);

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
