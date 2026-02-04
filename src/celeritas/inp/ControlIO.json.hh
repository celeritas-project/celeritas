//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/ControlIO.json.hh
//---------------------------------------------------------------------------//
#pragma once

#include <nlohmann/json.hpp>

#include "Control.hh"

namespace celeritas
{
namespace inp
{
//---------------------------------------------------------------------------//

void to_json(nlohmann::json& j, CoreStateCapacity const&);
void from_json(nlohmann::json const& j, CoreStateCapacity&);

void to_json(nlohmann::json& j, OpticalStateCapacity const&);
void from_json(nlohmann::json const& j, OpticalStateCapacity&);

void to_json(nlohmann::json& j, DeviceDebug const&);
void from_json(nlohmann::json const& j, DeviceDebug&);

void to_json(nlohmann::json& j, Control const&);
void from_json(nlohmann::json const& j, Control&);

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
