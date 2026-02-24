//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/DiagnosticsIO.json.cc
//---------------------------------------------------------------------------//
#include "DiagnosticsIO.json.hh"

#include "corecel/io/JsonUtils.json.hh"

namespace celeritas
{
namespace inp
{
//---------------------------------------------------------------------------//
//!@{
//! I/O routines for JSON
//! \todo Complete diagnostics IO

void to_json(nlohmann::json& j, Timers const& v)
{
    j = nlohmann::json{CELER_JSON_PAIR(v, action), CELER_JSON_PAIR(v, step)};
}

void from_json(nlohmann::json const& j, Timers& v)
{
    CELER_JSON_LOAD_OPTION(j, v, action);
    CELER_JSON_LOAD_OPTION(j, v, step);
}

//!@}

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
