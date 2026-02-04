//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/ControlIO.json.cc
//---------------------------------------------------------------------------//
#include "ControlIO.json.hh"

#include "corecel/io/JsonUtils.json.hh"

namespace celeritas
{
namespace inp
{
//---------------------------------------------------------------------------//
//!@{
//! I/O routines for JSON
//! \todo How should we set host/device specific default capacities?
//! \todo Revisit which capacity values are required/optional/defaulted

void to_json(nlohmann::json& j, CoreStateCapacity const& v)
{
    j = nlohmann::json{
        CELER_JSON_PAIR(v, primaries),
        CELER_JSON_PAIR(v, tracks),
        CELER_JSON_PAIR(v, initializers),
        CELER_JSON_PAIR_OPTIONAL(v, secondaries),
        CELER_JSON_PAIR_OPTIONAL(v, events),
    };
}

void from_json(nlohmann::json const& j, CoreStateCapacity& v)
{
    CELER_JSON_LOAD_REQUIRED(j, v, primaries);
    CELER_JSON_LOAD_REQUIRED(j, v, tracks);
    CELER_JSON_LOAD_REQUIRED(j, v, initializers);
    CELER_JSON_LOAD_OPTIONAL(j, v, secondaries);
    CELER_JSON_LOAD_OPTIONAL(j, v, events);
}

void to_json(nlohmann::json& j, OpticalStateCapacity const& v)
{
    j = nlohmann::json{
        CELER_JSON_PAIR(v, primaries),
        CELER_JSON_PAIR(v, tracks),
        CELER_JSON_PAIR(v, generators),
    };
}

void from_json(nlohmann::json const& j, OpticalStateCapacity& v)
{
    CELER_JSON_LOAD_REQUIRED(j, v, primaries);
    CELER_JSON_LOAD_REQUIRED(j, v, tracks);
    CELER_JSON_LOAD_REQUIRED(j, v, generators);
}

void to_json(nlohmann::json& j, DeviceDebug const& v)
{
    j = nlohmann::json{CELER_JSON_PAIR(v, sync_stream)};
}

void from_json(nlohmann::json const& j, DeviceDebug& v)
{
    CELER_JSON_LOAD_OPTION(j, v, sync_stream);
}

void to_json(nlohmann::json& j, Control const& v)
{
    j = nlohmann::json{
        CELER_JSON_PAIR(v, capacity),
        CELER_JSON_PAIR_OPTIONAL(v, optical_capacity),
        CELER_JSON_PAIR_OPTIONAL(v, track_order),
        CELER_JSON_PAIR_OPTIONAL(v, device_debug),
        CELER_JSON_PAIR(v, warm_up),
        CELER_JSON_PAIR(v, seed),
    };
}

void from_json(nlohmann::json const& j, Control& v)
{
    CELER_JSON_LOAD_REQUIRED(j, v, capacity);
    CELER_JSON_LOAD_OPTIONAL(j, v, optical_capacity);
    CELER_JSON_LOAD_OPTIONAL(j, v, track_order);
    CELER_JSON_LOAD_OPTIONAL(j, v, device_debug);
    CELER_JSON_LOAD_OPTION(j, v, warm_up);
    CELER_JSON_LOAD_OPTION(j, v, seed);
}

//!@}

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
