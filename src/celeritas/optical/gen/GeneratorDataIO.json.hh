//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/GeneratorDataIO.json.hh
//---------------------------------------------------------------------------//
#pragma once

#include <nlohmann/json.hpp>

#include "GeneratorData.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//

void to_json(nlohmann::json& j, GeneratorStepData const&);
void from_json(nlohmann::json const& j, GeneratorStepData&);

void to_json(nlohmann::json& j, GeneratorDistributionData const&);
void from_json(nlohmann::json const& j, GeneratorDistributionData&);

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
