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

#include "celeritas/geo/GeoFwd.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// Forward declarations
class CoreParams;
class CoreTrackView;
class ParticleTrackView;
class SimTrackView;

//---------------------------------------------------------------------------//
void to_json(nlohmann::json& j, CoreTrackView const&);

//---------------------------------------------------------------------------//
void to_json(nlohmann::json& j, GeoTrackView const&);

//---------------------------------------------------------------------------//
void to_json(nlohmann::json& j, ParticleTrackView const&);

//---------------------------------------------------------------------------//
void to_json(nlohmann::json& j, SimTrackView const&);

//---------------------------------------------------------------------------//
}  // namespace celeritas
