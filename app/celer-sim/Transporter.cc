//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-sim/Transporter.cc
//---------------------------------------------------------------------------//
#include "Transporter.hh"

#include <algorithm>
#include <csignal>
#include <memory>
#include <utility>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/data/Ref.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/ScopedTimeLog.hh"
#include "corecel/sys/ScopedSignalHandler.hh"
#include "celeritas/Types.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/Stepper.hh"
#include "celeritas/global/detail/ActionSequence.hh"
#include "celeritas/grid/VectorUtils.hh"
#include "celeritas/phys/Model.hh"

#include "StepTimer.hh"

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
//! Default virtual destructor
TransporterBase::~TransporterBase() = default;

//---------------------------------------------------------------------------//
/*!
 * Construct from persistent problem data.
 */
template<MemSpace M>
Transporter<M>::Transporter(TransporterInput inp)
    : max_steps_(inp.max_steps)
    , num_streams_(inp.params->max_streams())
    , store_track_counts_(inp.store_track_counts)
    , store_step_times_(inp.store_step_times)
{
    CELER_EXPECT(inp);

    // Create stepper
    CELER_LOG_LOCAL(status) << "Creating states";
    StepperInput step_input;
    step_input.params = inp.params;
    step_input.num_track_slots = inp.num_track_slots;
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
    StepperResult step_counts = (*stepper_)();
    CELER_ENSURE(step_counts.alive == 0);
}

//---------------------------------------------------------------------------//
/*!
 * Transport the input primaries and all secondaries produced.
 */
template<MemSpace M>
auto Transporter<M>::operator()(SpanConstPrimary primaries) -> TransporterResult
{
    // Initialize results
    TransporterResult result;
    auto append_track_counts = [&](StepperResult const& track_counts) {
        if (store_track_counts_)
        {
            result.initializers.push_back(track_counts.queued);
            result.active.push_back(track_counts.active);
            result.alive.push_back(track_counts.alive);
        }
        ++result.num_step_iterations;
        result.num_steps += track_counts.active;
        result.max_queued = std::max(result.max_queued, track_counts.queued);
    };

    constexpr size_type min_alloc{65536};
    result.initializers.reserve(std::min(min_alloc, max_steps_));
    result.active.reserve(std::min(min_alloc, max_steps_));
    result.alive.reserve(std::min(min_alloc, max_steps_));
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
    CELER_LOG_LOCAL(status)
        << "Transporting " << primaries.size() << " primaries";

    StepTimer record_step_time{store_step_times_ ? &result.step_times
                                                 : nullptr};
    size_type remaining_steps = max_steps_;

    auto& step = *stepper_;
    // Copy primaries to device and transport the first step
    auto track_counts = step(primaries);
    append_track_counts(track_counts);
    record_step_time();

    while (track_counts)
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
    }

    result.num_aborted = track_counts.alive + track_counts.queued;
    result.num_track_slots = stepper_->state().size();
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Transport the input primaries and all secondaries produced.
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
        auto const& action_ptrs = action_seq.actions();
        auto const& times = action_seq.accum_time();

        CELER_ASSERT(action_ptrs.size() == times.size());
        for (auto i : range(action_ptrs.size()))
        {
            (*result)[std::string{action_ptrs[i]->label()}] += times[i];
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
