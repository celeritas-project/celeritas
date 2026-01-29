//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/StandaloneInputIO.json.cc
//---------------------------------------------------------------------------//
#include "StandaloneInputIO.json.hh"

#include "corecel/io/JsonUtils.json.hh"
#include "celeritas/ext/GeantOpticalPhysicsOptionsIO.json.hh"

#include "ProblemIO.json.hh"
#include "SystemIO.json.hh"

namespace celeritas
{
namespace inp
{
//---------------------------------------------------------------------------//
static char const format_str[] = "optical-standalone-input";

//---------------------------------------------------------------------------//
//!@{
//! I/O routines for JSON
//! \todo Add JSON support for \c StandaloneInput

void to_json(nlohmann::json& j, OpticalStandaloneInput const& v)
{
    j = nlohmann::json{
        CELER_JSON_PAIR(v, system),
        CELER_JSON_PAIR(v, problem),
        CELER_JSON_PAIR(v, geant_setup),
    };

    save_format(j, format_str);
}

void from_json(nlohmann::json const& j, OpticalStandaloneInput& v)
{
    check_format(j, format_str);

    CELER_JSON_LOAD_OPTION(j, v, system);
    CELER_JSON_LOAD_REQUIRED(j, v, problem);
    CELER_JSON_LOAD_OPTION(j, v, geant_setup);
}

//!@}

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
