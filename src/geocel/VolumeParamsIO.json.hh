//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/VolumeParamsIO.json.hh
//---------------------------------------------------------------------------//
#pragma once

#include <iosfwd>
#include <nlohmann/json.hpp>

#include "geocel/inp/Model.hh"

#include "VolumeParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// Write volume hierarchy to JSON
void to_json(nlohmann::json& j, VolumeParams const& vp);

// Write volume hierarchy to a stream
std::ostream& operator<<(std::ostream& os, VolumeParams const& vp);

}  // namespace celeritas
