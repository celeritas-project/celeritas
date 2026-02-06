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
    };
}

void from_json(nlohmann::json const& j, CounterAccumStats& v)
{
    CELER_JSON_LOAD_REQUIRED(j, v, generators);
    CELER_JSON_LOAD_REQUIRED(j, v, steps);
    CELER_JSON_LOAD_REQUIRED(j, v, step_iters);
    CELER_JSON_LOAD_REQUIRED(j, v, flushes);
}

//!@}

//---------------------------------------------------------------------------//
}  // namespace celeritas
