//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/DeviceIO.json.hh
//---------------------------------------------------------------------------//
#pragma once

#include <nlohmann/json.hpp>

#include "Device.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//

// Write device diagnostics to JSON
void to_json(nlohmann::json& j, Device const& value);

//---------------------------------------------------------------------------//
}  // namespace celeritas
