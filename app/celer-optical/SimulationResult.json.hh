//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-optical/SimulationResult.json.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include <unordered_map>
#include <nlohmann/json.hpp>

#include "corecel/io/JsonUtils.json.hh"
#include "celeritas/phys/GeneratorCountersIO.json.hh"

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Timing results.
 *
 * \todo Add warmup time
 */
struct TimingResult
{
    using MapStrDouble = std::unordered_map<std::string, double>;

    double total{};  //!< Total transport time
    double setup{};  //!< One-time initialization cost
    MapStrDouble actions{};  //!< Accumulated action times
};

//---------------------------------------------------------------------------//
/*!
 * Results from transporting all tracks.
 */
struct SimulationResult
{
    TimingResult time{};
    CounterAccumStats counters{};
};

//---------------------------------------------------------------------------//

void to_json(nlohmann::json& j, TimingResult const& v)
{
    j = {
        CELER_JSON_PAIR(v, total),
        CELER_JSON_PAIR(v, setup),
        CELER_JSON_PAIR(v, actions),
    };
}

void to_json(nlohmann::json& j, SimulationResult const& v)
{
    j = {
        CELER_JSON_PAIR(v, time),
        CELER_JSON_PAIR(v, counters),
    };
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
