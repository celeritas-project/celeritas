//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/TimerOutput.cc
//---------------------------------------------------------------------------//
#include "TimerOutput.hh"

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/io/JsonPimpl.hh"
#include "celeritas/ext/GeantUtils.hh"

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
 */
TimerOutput::TimerOutput(size_type num_threads)
{
    CELER_EXPECT(num_threads > 0);

    action_time_.resize(num_threads);
    event_time_.resize(num_threads);
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

    obj = {
        {"_index", "thread"},
        {"actions", action_time_},
        {"events", event_time_},
        {"total", total_time_},
        {"setup", setup_time_},
    };

    j->obj = std::move(obj);
#else
    CELER_DISCARD(j);
#endif
}

//---------------------------------------------------------------------------//
/*!
 * Record the accumulated action times.
 */
void TimerOutput::RecordActionTime(MapStrReal&& time)
{
    size_type thread_id = get_geant_thread_id();
    CELER_ASSERT(thread_id < action_time_.size());
    action_time_[thread_id] = std::move(time);
}

//---------------------------------------------------------------------------//
/*!
 * Record the time for the event.
 */
void TimerOutput::RecordEventTime(real_type time)
{
    size_type thread_id = get_geant_thread_id();
    CELER_ASSERT(thread_id < event_time_.size());
    event_time_[thread_id].push_back(time);
}

//---------------------------------------------------------------------------//
/*!
 * Record the time for setup.
 *
 * This should be called once by the master thread.
 */
void TimerOutput::RecordSetupTime(real_type time)
{
    setup_time_ = time;
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
}  // namespace app
}  // namespace celeritas
