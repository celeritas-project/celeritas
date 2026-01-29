//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/SystemIO.json.hh
//---------------------------------------------------------------------------//
#pragma once

#include <nlohmann/json.hpp>

#include "System.hh"

namespace celeritas
{
namespace inp
{
//---------------------------------------------------------------------------//

void to_json(nlohmann::json& j, Device const&);
void from_json(nlohmann::json const& j, Device&);

void to_json(nlohmann::json& j, System const&);
void from_json(nlohmann::json const& j, System&);

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
