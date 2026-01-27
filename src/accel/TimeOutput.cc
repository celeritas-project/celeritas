//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/TimeOutput.cc
//---------------------------------------------------------------------------//
#include "TimeOutput.hh"

#include <nlohmann/json.hpp>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/io/JsonPimpl.hh"
#include "geocel/GeantUtils.hh"
#include "celeritas/Quantities.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with number of threads.
 */
TimeOutput::TimeOutput(size_type num_threads)
{
    CELER_EXPECT(num_threads > 0);

    action_time_.resize(num_threads);
    event_time_.resize(num_threads);
}

//---------------------------------------------------------------------------//
/*!
 * Write output to the given JSON object.
 */
void TimeOutput::output(JsonPimpl* j) const
{
    using json = nlohmann::json;
    using TimeSecond = RealQuantity<units::Second>;

    auto obj = json::object();

    obj = {
        {"_units", TimeSecond::unit_type::label()},
        {"_index", "thread"},
        {"actions", action_time_},
        {"events", event_time_},
        {"total", total_time_},
        {"setup", setup_time_},
    };

    j->obj = std::move(obj);
}

//---------------------------------------------------------------------------//
/*!
 * Record the accumulated action times.
 */
void TimeOutput::RecordActionTime(MapStrDbl&& time)
{
    size_type thread_id = get_geant_thread_id();
    CELER_ASSERT(thread_id < action_time_.size());
    action_time_[thread_id] = std::move(time);
}

//---------------------------------------------------------------------------//
/*!
 * Record the time for the event.
 */
void TimeOutput::RecordEventTime(double time)
{
    size_type thread_id = get_geant_thread_id();
    CELER_ASSERT(thread_id < event_time_.size());
    event_time_[thread_id].push_back(time);
}

//---------------------------------------------------------------------------//
/*!
 * Record the time for setting up Celeritas.
 *
 * This should be called once by the main thread.
 */
void TimeOutput::RecordSetupTime(double time)
{
    setup_time_ = time;
}

//---------------------------------------------------------------------------//
/*!
 * Record the total time spent in transport and hit I/O (excluding setup).
 *
 * This should be called once by the main thread.
 */
void TimeOutput::RecordTotalTime(double time)
{
    total_time_ = time;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
