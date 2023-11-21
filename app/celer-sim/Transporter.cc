//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-sim/Transporter.cc
//---------------------------------------------------------------------------//
#include "Transporter.hh"

#include <csignal>
#include <memory>
#include <utility>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/data/Ref.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/ScopedTimeLog.hh"
#include "corecel/sys/ScopedSignalHandler.hh"
#include "corecel/sys/Stopwatch.hh"
#include "celeritas/Types.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/Stepper.hh"
#include "celeritas/global/detail/ActionSequence.hh"
#include "celeritas/grid/VectorUtils.hh"
#include "celeritas/phys/Model.hh"

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
{
    CELER_EXPECT(inp);

    // Create stepper
    CELER_LOG_LOCAL(status) << "Creating states";
    StepperInput step_input;
    step_input.params = inp.params;
    step_input.num_track_slots = inp.num_track_slots;
    step_input.stream_id = inp.stream_id;
    step_input.sync = inp.sync;
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
auto Transporter<M>::operator()(SpanConstPrimary primaries)
    -> TransporterResult
{
    // Initialize results
    TransporterResult result;
    auto append_track_counts = [&result](StepperResult const& track_counts) {
        result.initializers.push_back(track_counts.queued);
        result.active.push_back(track_counts.active);
        result.alive.push_back(track_counts.alive);
    };

    // Abort cleanly for interrupt and user-defined signals
#ifndef _WIN32
    ScopedSignalHandler interrupted{SIGINT, SIGUSR2};
#else
    ScopedSignalHandler interrupted{SIGINT};
#endif
    CELER_LOG_LOCAL(status)
        << "Transporting " << primaries.size() << " primaries";

    Stopwatch get_step_time;
    size_type remaining_steps = max_steps_;

    auto& step = *stepper_;
    // Copy primaries to device and transport the first step
    auto track_counts = step(primaries);
    if (store_track_counts_)
    {
        append_track_counts(track_counts);
    }
    if (num_streams_ == 1)
    {
        result.step_times.push_back(get_step_time());
    }

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
            CELER_LOG_LOCAL(error)
                << "Caught interrupt signal: aborting transport "
                   "loop";
            interrupted = {};
            break;
        }

        get_step_time = {};
        track_counts = step();
        if (store_track_counts_)
        {
            append_track_counts(track_counts);
        }
        if (num_streams_ == 1)
        {
            result.step_times.push_back(get_step_time());
        }
    }

    result.num_track_slots = stepper_->state().size();
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Transport the input primaries and all secondaries produced.
 */
template<MemSpace M>
auto Transporter<M>::get_action_times() const -> MapStrReal
{
    // Get kernel timing if running with a single stream and if either on the
    // device with synchronization enabled or on the host
    MapStrReal result;
    auto const& step = *stepper_;
    auto const& action_seq = step.actions();
    if (num_streams_ == 1 && (M == MemSpace::host || action_seq.sync()))
    {
        auto const& action_ptrs = action_seq.actions();
        auto const& times = action_seq.accum_time();

        CELER_ASSERT(action_ptrs.size() == times.size());
        for (auto i : range(action_ptrs.size()))
        {
            auto&& label = action_ptrs[i]->label();
            result[label] = times[i];
        }
    }
    return result;
}

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//

template class Transporter<MemSpace::host>;
template class Transporter<MemSpace::device>;

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
