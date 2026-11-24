//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/VolumeParamsIO.json.hh
//---------------------------------------------------------------------------//
#pragma once

#include <nlohmann/json.hpp>

#include "VolumeParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// Write volume hierarchy to JSON
void to_json(nlohmann::json& j, VolumeParams const& vp);

}  // namespace celeritas
