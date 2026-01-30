//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/ProblemIO.json.cc
//---------------------------------------------------------------------------//
#include "ProblemIO.json.hh"

#include "corecel/io/JsonUtils.json.hh"
#include "geocel/inp/ModelIO.json.hh"

#include "ControlIO.json.hh"
#include "DiagnosticsIO.json.hh"
#include "EventsIO.json.hh"
#include "TrackingIO.json.hh"

namespace celeritas
{
namespace inp
{
//---------------------------------------------------------------------------//
//!@{
//! I/O routines for JSON
//! \todo Add JSON support for \c Problem
//! \todo Add JSON support for \c OpticalPhysics when it's used as input

void to_json(nlohmann::json& j, OpticalProblem const& v)
{
    j = nlohmann::json{
        CELER_JSON_PAIR(v, model),
        CELER_JSON_PAIR(v, generator),
        CELER_JSON_PAIR(v, limits),
        CELER_JSON_PAIR(v, capacity),
        CELER_JSON_PAIR(v, seed),
        CELER_JSON_PAIR(v, timers),
        CELER_JSON_PAIR_WHEN(v, perfetto_file, !v.perfetto_file.empty()),
        CELER_JSON_PAIR(v, output_file),
    };
}

void from_json(nlohmann::json const& j, OpticalProblem& v)
{
    CELER_JSON_LOAD_REQUIRED(j, v, model);
    CELER_JSON_LOAD_REQUIRED(j, v, generator);
    CELER_JSON_LOAD_OPTION(j, v, limits);
    CELER_JSON_LOAD_REQUIRED(j, v, capacity);
    CELER_JSON_LOAD_OPTION(j, v, seed);
    CELER_JSON_LOAD_OPTION(j, v, timers);
    CELER_JSON_LOAD_OPTION(j, v, perfetto_file);
    CELER_JSON_LOAD_OPTION(j, v, output_file);
}

//!@}

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
