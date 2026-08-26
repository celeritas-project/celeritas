//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file larceler/LarRunnerDiagnosticsIO.json.hh
//---------------------------------------------------------------------------//
#pragma once

#include <nlohmann/json.hpp>

#include "corecel/io/JsonUtils.json.hh"
#include "celeritas/phys/GeneratorCountersIO.json.hh"  // IWYU pragma: keep

#include "LarRunnerDiagnostics.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//

void to_json(nlohmann::json& j, LarTiming const& v)
{
    j = {
        CELER_JSON_PAIR(v, setup),
        CELER_JSON_PAIR(v, run),
        CELER_JSON_PAIR(v, teardown),
        CELER_JSON_PAIR(v, actions),
        CELER_JSON_PAIR(v, steps),
    };
}

void to_json(nlohmann::json& j, LarRunnerDiagnostics const& v)
{
    j = {
        CELER_JSON_PAIR(v, time),
        CELER_JSON_PAIR(v, counters),
    };
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
