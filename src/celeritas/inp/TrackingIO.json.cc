//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/TrackingIO.json.cc
//---------------------------------------------------------------------------//
#include "TrackingIO.json.hh"

#include "corecel/io/JsonUtils.json.hh"

namespace celeritas
{
namespace inp
{
//---------------------------------------------------------------------------//
//!@{
//! I/O routines for JSON

void to_json(nlohmann::json& j, CoreTrackingLimits const& v)
{
    j = nlohmann::json{
        CELER_JSON_PAIR(v, steps),
        CELER_JSON_PAIR(v, step_iters),
        CELER_JSON_PAIR(v, field_substeps),
    };
}

void from_json(nlohmann::json const& j, CoreTrackingLimits& v)
{
    CELER_JSON_LOAD_OPTION(j, v, steps);
    CELER_JSON_LOAD_OPTION(j, v, step_iters);
    CELER_JSON_LOAD_OPTION(j, v, field_substeps);
}

void to_json(nlohmann::json& j, OpticalTrackingLimits const& v)
{
    j = nlohmann::json{
        CELER_JSON_PAIR(v, steps),
        CELER_JSON_PAIR(v, step_iters),
    };
}

void from_json(nlohmann::json const& j, OpticalTrackingLimits& v)
{
    CELER_JSON_LOAD_OPTION(j, v, steps);
    CELER_JSON_LOAD_OPTION(j, v, step_iters);
}

void to_json(nlohmann::json& j, Tracking const& v)
{
    j = nlohmann::json{
        CELER_JSON_PAIR(v, limits),
        CELER_JSON_PAIR(v, optical_limits),
        CELER_JSON_PAIR(v, force_step_limit),
    };
}

void from_json(nlohmann::json const& j, Tracking& v)
{
    CELER_JSON_LOAD_OPTION(j, v, limits);
    CELER_JSON_LOAD_OPTION(j, v, optical_limits);
    CELER_JSON_LOAD_OPTION(j, v, force_step_limit);
}

//!@}

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
