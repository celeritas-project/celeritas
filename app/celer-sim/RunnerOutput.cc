//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-sim/RunnerOutput.cc
//---------------------------------------------------------------------------//
#include "RunnerOutput.hh"

#include <utility>

#include "celeritas_config.h"
#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/io/JsonPimpl.hh"
#include "celeritas/Types.hh"

#if CELERITAS_USE_JSON
#    include <nlohmann/json.hpp>

#    include "corecel/io/LabelIO.json.hh"
#endif

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Construct from simulation result.
 */
RunnerOutput::RunnerOutput(SimulationResult result)
    : result_(std::move(result))
{
}

//---------------------------------------------------------------------------//
/*!
 * Write output to the given JSON object.
 */
void RunnerOutput::output(JsonPimpl* j) const
{
#if CELERITAS_USE_JSON
    using json = nlohmann::json;

    auto obj = json::object();

    auto active = json::array();
    auto alive = json::array();
    auto initializers = json::array();
    auto num_track_slots = json::array();
    auto num_step_iterations = json::array();
    auto num_steps = json::array();
    auto num_aborted = json::array();
    auto max_queued = json::array();
    auto step_times = json::array();

    for (auto const& event : result_.events)
    {
        if (!event.active.empty())
        {
            active.push_back(event.active);
            alive.push_back(event.alive);
            initializers.push_back(event.initializers);
        }
        num_track_slots.push_back(event.num_track_slots);
        num_step_iterations.push_back(event.num_step_iterations);
        num_steps.push_back(event.num_steps);
        num_aborted.push_back(event.num_aborted);
        max_queued.push_back(event.max_queued);
        if (!event.step_times.empty())
        {
            step_times.push_back(event.step_times);
        }
    }
    obj["_index"] = {"event", "step"};
    obj["active"] = std::move(active);
    obj["alive"] = std::move(alive);
    obj["initializers"] = std::move(initializers);
    obj["num_track_slots"] = std::move(num_track_slots);
    obj["num_step_iterations"] = std::move(num_step_iterations);
    obj["num_steps"] = std::move(num_steps);
    obj["num_aborted"] = std::move(num_aborted);
    obj["max_queued"] = std::move(max_queued);
    obj["time"] = {
        {"steps", std::move(step_times)},
        {"actions", result_.action_times},
        {"total", result_.total_time},
        {"setup", result_.setup_time},
        {"warmup", result_.warmup_time},
    };
    obj["num_streams"] = result_.num_streams;

    j->obj = std::move(obj);
#else
    CELER_DISCARD(j);
#endif
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
