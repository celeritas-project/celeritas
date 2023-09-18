//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/TimerOutput.cc
//---------------------------------------------------------------------------//
#include "TimerOutput.hh"

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/io/JsonPimpl.hh"

#if CELERITAS_USE_JSON
#    include <nlohmann/json.hpp>
#endif

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Construct with number of threads.
 *
 * Accumulated action times and per-event times are only collected in
 * single-threaded mode.
 */
TimerOutput::TimerOutput(size_type num_threads)
    : detailed_timing_{num_threads == 1}
{
}

//---------------------------------------------------------------------------//
/*!
 * Write output to the given JSON object.
 */
void TimerOutput::output(JsonPimpl* j) const
{
#if CELERITAS_USE_JSON
    using json = nlohmann::json;

    auto obj = json::object();

    obj["time"] = {
        {"actions", action_time_},
        {"events", event_time_},
        {"total", total_time_},
    };

    j->obj = std::move(obj);
#else
    (void)sizeof(j);
#endif
}

//---------------------------------------------------------------------------//
/*!
 * Record the total time spent in transport and hit I/O (excluding setup).
 *
 * This should be called once by the master thread.
 */
void TimerOutput::RecordTotalTime(real_type time)
{
    total_time_ = time;
}

//---------------------------------------------------------------------------//
/*!
 * Record the accumulated action times.
 */
void TimerOutput::RecordActionTime(MapStrReal&& time)
{
    if (detailed_timing_)
    {
        action_time_ = std::move(time);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Record the time for the event.
 */
void TimerOutput::RecordEventTime(real_type time)
{
    if (detailed_timing_)
    {
        event_time_.push_back(time);
    }
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
