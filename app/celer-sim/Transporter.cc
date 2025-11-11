//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-sim/Transporter.cc
//---------------------------------------------------------------------------//
#include "Transporter.hh"

#include <algorithm>
#include <csignal>
#include <memory>
#include <numeric>
#include <utility>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/data/CollectionAlgorithms.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/ScopedTimeLog.hh"
#include "corecel/sys/ScopedSignalHandler.hh"
#include "corecel/sys/TraceCounter.hh"
#include "celeritas/Types.hh"
#include "celeritas/global/ActionSequence.hh"
#include "celeritas/global/CoreParams.hh"  // IWYU pragma: keep
#include "celeritas/global/Stepper.hh"
#include "celeritas/optical/OpticalCollector.hh"  // IWYU pragma: keep
#include "celeritas/phys/GeneratorCounters.hh"
#include "celeritas/phys/Model.hh"

#include "StepTimer.hh"

namespace celeritas
{
namespace app
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Log progress after N events when requested.
 */
void log_progress(EventId const id, size_type const num_primaries)
{
    CELER_EXPECT(id);
    CELER_EXPECT(num_primaries > 0);
    CELER_LOG_LOCAL(status)
        << "Event " << id.unchecked_get() << ": transporting " << num_primaries
        << (num_primaries == 1 ? " primary" : " primaries");
}
}  // namespace

//---------------------------------------------------------------------------//
//! Default virtual destructor
TransporterBase::~TransporterBase() = default;

//---------------------------------------------------------------------------//
/*!
 * Construct from persistent problem data.
 */
template<MemSpace M>
Transporter<M>::Transporter(TransporterInput inp)
    : optical_{std::move(inp.optical)}
    , max_steps_(inp.max_steps)
    , num_streams_(inp.params->max_streams())
    , log_progress_(inp.log_progress)
    , store_track_counts_(inp.store_track_counts)
    , store_step_times_(inp.store_step_times)
{
    CELER_EXPECT(inp);
    CELER_VALIDATE(log_progress_ > 0, << "log_progress must be positive");

    // Create stepper
    CELER_LOG_LOCAL(status) << "Creating states";
    StepperInput step_input;
    step_input.params = inp.params;
    step_input.stream_id = inp.stream_id;
    step_input.action_times = inp.action_times;
    stepper_ = std::make_shared<Stepper<M>>(std::move(step_input));
}

//---------------------------------------------------------------------------//
/*!
 * Run a single step with no active states to "warm up".
 *
 * This is to reduce the uncertainty in timing for problems, especially on AMD
 * hardware.
 */
template<MemSpace M>
void Transporter<M>::operator()()
{
    CELER_LOG(status) << "Warming up";
    ScopedTimeLog scoped_time;
    stepper_->warm_up();
}

//---------------------------------------------------------------------------//
/*!
 * Transport the input primaries and all secondaries produced.
 */
template<MemSpace M>
auto Transporter<M>::operator()(SpanConstPrimary primaries)
    -> TransporterResult
{
    // Initialize results
    TransporterResult result;
    auto append_track_counts = [&](StepperResult const& track_counts) {
        if (store_track_counts_)
        {
            result.generated.push_back(track_counts.generated);
            result.initializers.push_back(track_counts.queued);
            result.active.push_back(track_counts.active);
            result.alive.push_back(track_counts.alive);
            if constexpr (M == MemSpace::host)
            {
                auto stream_id
                    = std::to_string(stepper_->state_ref().stream_id.get());
                trace_counter(std::string("active-" + stream_id).c_str(),
                              track_counts.active);
                trace_counter(std::string("alive-" + stream_id).c_str(),
                              track_counts.alive);
                trace_counter(std::string("dead-" + stream_id).c_str(),
                              track_counts.active - track_counts.alive);
                trace_counter(std::string("queued-" + stream_id).c_str(),
                              track_counts.queued);
            }
        }
        ++result.num_step_iterations;
        result.num_steps += track_counts.active;
        result.max_queued = std::max(result.max_queued, track_counts.queued);
    };

    constexpr size_type min_alloc{65536};
    if (store_track_counts_)
    {
        result.generated.reserve(std::min(min_alloc, max_steps_));
        result.initializers.reserve(std::min(min_alloc, max_steps_));
        result.active.reserve(std::min(min_alloc, max_steps_));
        result.alive.reserve(std::min(min_alloc, max_steps_));
    }
    if (store_step_times_)
    {
        result.step_times.reserve(std::min(min_alloc, max_steps_));
    }

    // Abort cleanly for interrupt and user-defined signals
#ifndef _WIN32
    ScopedSignalHandler interrupted{SIGINT, SIGUSR2};
#else
    ScopedSignalHandler interrupted{SIGINT};
#endif

    if (auto const evt_id = primaries.front().event_id;
        evt_id.get() % log_progress_ == 0)
    {
        log_progress(evt_id, primaries.size());
    }

    StepTimer record_step_time{store_step_times_ ? &result.step_times
                                                 : nullptr};
    size_type remaining_steps = max_steps_;

    auto& step = *stepper_;
    // Copy primaries to device and transport the first step
    auto track_counts = step(primaries);
    append_track_counts(track_counts);
    record_step_time();

    GeneratorCounters optical_counts;
    while (track_counts || !optical_counts.empty())
    {
        if (CELER_UNLIKELY(--remaining_steps == 0))
        {
            CELER_LOG_LOCAL(error) << "Exceeded step count of " << max_steps_
                                   << ": aborting transport loop";
            break;
        }
        if (CELER_UNLIKELY(interrupted()))
        {
            CELER_LOG_LOCAL(error) << "Caught interrupt signal: aborting "
                                      "transport loop";
            interrupted = {};
            break;
        }

        track_counts = step();
        append_track_counts(track_counts);
        record_step_time();

        if (optical_)
        {
            auto& aux = stepper_->sp_state()->aux();
            optical_counts = optical_->buffer_counts(aux);
        }
    }

    auto counters = copy_to_host(stepper_->state_ref().init.track_counters);
    result.num_tracks = std::accumulate(counters.data().get(),
                                        counters.data().get() + counters.size(),
                                        size_type(0));
    result.num_aborted = track_counts.alive + track_counts.queued;
    result.num_track_slots = stepper_->state().size();

    if (result.num_aborted > 0)
    {
        // Reset the state data for the next event if the stepping loop was
        // aborted early
        step.reset_state();
    }

    // Get optical counters
    if (optical_)
    {
        auto& aux = stepper_->sp_state()->aux();
        auto counters = optical_->exchange_counters(aux);

        OpticalCounts oc;
        for (auto const& gen : counters.generators)
        {
            oc.tracks += gen.num_generated;
            oc.generators += gen.buffer_size;
        }
        oc.steps = counters.steps;
        oc.step_iters = counters.step_iters;
        oc.flushes = counters.flushes;

        CELER_LOG_LOCAL(debug)
            << "Tracked " << oc.tracks << " photons from " << oc.generators
            << " distributions for " << oc.steps << " steps, using "
            << counters.step_iters << " step iterations over " << oc.flushes
            << " flushes";

        auto const& buffer_counts = optical_->buffer_counts(aux);
        if (!buffer_counts.empty())
        {
            CELER_LOG_LOCAL(warning)
                << "Not all optical photons were tracked "
                   "at the end of the stepping loop: "
                << buffer_counts.num_pending << " queued photons from "
                << buffer_counts.buffer_size << " distributions";
        }

        result.num_optical = std::move(oc);
    }

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Merge times across all threads.
 *
 * \todo Action times are to be refactored as aux data.
 */
template<MemSpace M>
void Transporter<M>::accum_action_times(MapStrDouble* result) const
{
    // Get kernel timing if running with a single stream and if
    // synchronization is enabled
    auto const& step = *stepper_;
    auto const& action_seq = step.actions();
    if (action_seq.action_times())
    {
        auto const& action_ptrs = action_seq.actions().step();
        auto const& times = action_seq.accum_time();

        CELER_ASSERT(action_ptrs.size() == times.size());
        for (auto i : range(action_ptrs.size()))
        {
            (*result)[std::string{action_ptrs[i]->label()}] += times[i];
        }

        if (optical_)
        {
            // Save optical loop action times
            auto optical_times = optical_->get_action_times(step.state().aux());
            for (auto&& [label, time] : optical_times)
            {
                // Prefix label to distinguish from core actions
                (*result)["optical::" + label] += time;
            }
        }
    }
}

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//

template class Transporter<MemSpace::host>;
template class Transporter<MemSpace::device>;

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
