//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/StandaloneInputIO.json.cc
//---------------------------------------------------------------------------//
#include "StandaloneInputIO.json.hh"

#include "corecel/io/JsonUtils.json.hh"
#include "celeritas/ext/GeantOpticalPhysicsOptionsIO.json.hh"

#include "EventsIO.json.hh"
#include "ImportIO.json.hh"
#include "ProblemIO.json.hh"
#include "SystemIO.json.hh"

namespace celeritas
{
namespace inp
{
//---------------------------------------------------------------------------//
static char const format_str[] = "standalone-input";
static char const optical_format_str[] = "optical-standalone-input";

//---------------------------------------------------------------------------//
//!@{
//! I/O routines for JSON

void to_json(nlohmann::json& j, StandaloneInput const& v)
{
    j = nlohmann::json{
        CELER_JSON_PAIR(v, system),
        CELER_JSON_PAIR(v, problem),
        CELER_JSON_PAIR_OPTIONAL(v, geant_setup),
        CELER_JSON_PAIR(v, physics_import),
        CELER_JSON_PAIR(v, events),
    };

    save_format(j, format_str);
}

void from_json(nlohmann::json const& j, StandaloneInput& v)
{
    check_format(j, format_str);

    CELER_JSON_LOAD_OPTION(j, v, system);

    // Set default values that depend on whether the device is enabled
    bool const use_device = v.system.device.has_value();
    v.problem.control.capacity = CoreStateCapacity::from_default(use_device);
    if (use_device)
    {
        v.problem.control.warm_up = true;
        v.problem.control.track_order = TrackOrder::init_charge;
    }

    CELER_JSON_LOAD_REQUIRED(j, v, problem);
    CELER_JSON_LOAD_OPTIONAL(j, v, geant_setup);
    CELER_JSON_LOAD_OPTION(j, v, physics_import);
    CELER_JSON_LOAD_REQUIRED(j, v, events);
}

void to_json(nlohmann::json& j, OpticalStandaloneInput const& v)
{
    j = nlohmann::json{
        CELER_JSON_PAIR(v, system),
        CELER_JSON_PAIR(v, problem),
        CELER_JSON_PAIR(v, geant_setup),
    };

    save_format(j, optical_format_str);
}

void from_json(nlohmann::json const& j, OpticalStandaloneInput& v)
{
    check_format(j, optical_format_str);

    CELER_JSON_LOAD_OPTION(j, v, system);
    CELER_JSON_LOAD_REQUIRED(j, v, problem);
    CELER_JSON_LOAD_OPTION(j, v, geant_setup);
}

//!@}

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
