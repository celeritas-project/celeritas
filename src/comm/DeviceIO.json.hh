//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file DeviceIO.json.hh
//---------------------------------------------------------------------------//
#pragma once

#include <nlohmann/json.hpp>
#include "Device.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//

// Write device diagnostics to JSON
void to_json(nlohmann::json& j, const Device& value);

//---------------------------------------------------------------------------//
} // namespace celeritas
