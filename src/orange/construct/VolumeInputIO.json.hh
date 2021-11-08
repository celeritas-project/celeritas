//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file VolumeInputIO.json.hh
//---------------------------------------------------------------------------//
#pragma once

#include <nlohmann/json.hpp>
#include "VolumeInput.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//

void from_json(const nlohmann::json& j, VolumeInput& value);

//---------------------------------------------------------------------------//
} // namespace celeritas
