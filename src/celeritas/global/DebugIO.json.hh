//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/DebugIO.json.hh
//! \brief Write *on-host* track views to JSON for debugging.
//---------------------------------------------------------------------------//
#pragma once

#include <nlohmann/json.hpp>

namespace celeritas
{
class SimTrackView;

//---------------------------------------------------------------------------//
void to_json(nlohmann::json& j, SimTrackView const&);

//---------------------------------------------------------------------------//
}  // namespace celeritas
