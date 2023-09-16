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
 * Construct with number of events.
 */
TimerOutput::TimerOutput(size_type num_events)
{
    CELER_EXPECT(num_events > 0);

    event_time_.resize(num_events);
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
        //{"actions", std::move(action_times)},
        {"events", event_time_},
        {"total", total_time_},
        //{"setup", result_.setup_time},
    };

    j->obj = std::move(obj);
#else
    (void)sizeof(j);
#endif
}

//---------------------------------------------------------------------------//
/*!
 * Record the total time for the run.
 */
void TimerOutput::RecordTotalTime(real_type time)
{
    total_time_ = time;
}

//---------------------------------------------------------------------------//
/*!
 * Record the time for the given event.
 */
void TimerOutput::RecordEventTime(G4Event const* event, real_type time)
{
    CELER_EXPECT(event);

    auto event_id = event->GetEventID();
    CELER_ASSERT(event_id < static_cast<int>(event_time_.size()));
    event_time_[event_id] = time;
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
