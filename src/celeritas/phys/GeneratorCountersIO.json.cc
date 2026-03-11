//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/GeneratorCountersIO.json.cc
//---------------------------------------------------------------------------//
#include "GeneratorCountersIO.json.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//!@{
//! I/O routines for JSON

void to_json(nlohmann::json& j, CounterAccumStats const& v)
{
    j = nlohmann::json{
        CELER_JSON_PAIR(v, generators),
        CELER_JSON_PAIR(v, steps),
        CELER_JSON_PAIR(v, step_iters),
        CELER_JSON_PAIR(v, flushes),
        CELER_JSON_PAIR(v, num_cut),
        CELER_JSON_PAIR(v, num_errored),
    };
}

void from_json(nlohmann::json const& j, CounterAccumStats& v)
{
    CELER_JSON_LOAD_REQUIRED(j, v, generators);
    CELER_JSON_LOAD_REQUIRED(j, v, steps);
    CELER_JSON_LOAD_REQUIRED(j, v, step_iters);
    CELER_JSON_LOAD_REQUIRED(j, v, flushes);
    if (j.contains("num_cut"))
        j.at("num_cut").get_to(v.num_cut);
    if (j.contains("num_errored"))
        j.at("num_errored").get_to(v.num_errored);
}

//!@}

//---------------------------------------------------------------------------//
}  // namespace celeritas
