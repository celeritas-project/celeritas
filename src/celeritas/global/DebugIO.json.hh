//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/DebugIO.json.hh
//! \brief Write *on-host* track views to JSON for debugging.
//---------------------------------------------------------------------------//
#pragma once

#include <nlohmann/json.hpp>

#include "celeritas/geo/GeoFwd.hh"
#if CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_ORANGE
#    include "orange/DebugIO.json.hh"
#endif

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
#if CELERITAS_CORE_GEO != CELERITAS_CORE_GEO_ORANGE
void to_json(nlohmann::json& j, GeoTrackView const&);
#endif

//---------------------------------------------------------------------------//
void to_json(nlohmann::json& j, ParticleTrackView const&);

//---------------------------------------------------------------------------//
void to_json(nlohmann::json& j, SimTrackView const&);

//---------------------------------------------------------------------------//
}  // namespace celeritas
