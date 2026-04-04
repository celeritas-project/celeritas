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
//!@}

//---------------------------------------------------------------------------//
}  // namespace celeritas
