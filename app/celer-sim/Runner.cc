//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-sim/Runner.cc
//---------------------------------------------------------------------------//
#include "Runner.hh"

#include <memory>
#include <utility>
#include <vector>

#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"
#include "corecel/sys/ThreadId.hh"

#ifdef _OPENMP
#    include <omp.h>
#endif

#include "corecel/sys/Device.hh"
#include "celeritas/Types.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/inp/StandaloneInput.hh"
#include "celeritas/setup/StandaloneInput.hh"

#include "RunnerInput.hh"
#include "Transporter.hh"

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Construct on all threads from a JSON input and shared output manager.
 */
Runner::Runner(RunnerInput const& old_inp)
{
    // Convert to new format and set up problem
    inp::StandaloneInput si = to_input(old_inp);
    auto loaded = setup::standalone_input(si);
    core_params_ = std::move(loaded.problem.core_params);
    CELER_ASSERT(core_params_);
    events_ = std::move(loaded.events);

    if (old_inp.merge_events)
    {
        // Merge all events into a single one
        VecPrimary merged;

        // Reserve space in the merged vector
        merged.reserve(std::accumulate(
            events_.begin(),
            events_.end(),
            std::size_t{0},
            [](std::size_t sum, auto const& v) { return sum + v.size(); }));

        for (auto const& v : events_)
        {
            merged.insert(merged.end(), v.begin(), v.end());
        }
        events_ = {std::move(merged)};
    }

    use_device_ = old_inp.use_device;

    CELER_VALIDATE(old_inp.max_steps > 0,
                   << "nonpositive max_steps=" << old_inp.max_steps);
    transporter_input_ = std::make_shared<TransporterInput>();
    transporter_input_->optical = std::move(loaded.problem.optical_collector);
    transporter_input_->params = core_params_;

    transporter_input_->max_steps = old_inp.max_steps;
    transporter_input_->store_track_counts = old_inp.write_track_counts;
    transporter_input_->store_step_times = old_inp.write_step_times;
    transporter_input_->action_times = old_inp.action_times;
    transporter_input_->log_progress = old_inp.log_progress;

    transporters_.resize(this->num_streams());

    CELER_ENSURE(core_params_);
}

//---------------------------------------------------------------------------//
/*!
 * Run a single step with no active states to "warm up".
 *
 * This is to reduce the uncertainty in timing for problems, especially on AMD
 * hardware.
 */
void Runner::warm_up()
{
    auto& transport = this->get_transporter(StreamId{0});
    transport();
}

//---------------------------------------------------------------------------//
/*!
 * Run on a single stream/thread, returning the transport result.
 *
 * This will partition the input primaries among all the streams.
 */
auto Runner::operator()(StreamId stream, EventId event) -> RunnerResult
{
    CELER_EXPECT(stream < this->num_streams());
    CELER_EXPECT(event < this->num_events());

    auto& transport = this->get_transporter(stream);
    return transport(make_span(events_[event.get()]));
}

//---------------------------------------------------------------------------//
/*!
 * Run all events simultaneously on a single stream.
 */
auto Runner::operator()() -> RunnerResult
{
    CELER_EXPECT(events_.size() == 1);
    CELER_EXPECT(this->num_streams() == 1);

    auto& transport = this->get_transporter(StreamId{0});
    return transport(make_span(events_.front()));
}

//---------------------------------------------------------------------------//
/*!
 * Number of streams supported.
 */
StreamId::size_type Runner::num_streams() const
{
    CELER_EXPECT(core_params_);
    return core_params_->max_streams();
}

//---------------------------------------------------------------------------//
/*!
 * Total number of events.
 */
size_type Runner::num_events() const
{
    return events_.size();
}

//---------------------------------------------------------------------------//
/*!
 * Get the accumulated action times.
 *
 * This is a *mean* value over all streams.
 *
 * \todo Refactor action times gathering: see celeritas::ActionSequence .
 */
auto Runner::get_action_times() const -> MapStrDouble
{
    MapStrDouble result;
    size_type num_streams{0};
    for (auto sid : range(StreamId{this->num_streams()}))
    {
        if (auto* transport = this->get_transporter_ptr(sid))
        {
            transport->accum_action_times(&result);
            ++num_streams;
        }
    }

    double norm{1 / static_cast<double>(num_streams)};
    for (auto&& [action, time] : result)
    {
        time *= norm;
    }

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Get the transporter for the given stream, constructing if necessary.
 */
auto Runner::get_transporter(StreamId stream) -> TransporterBase&
{
    CELER_EXPECT(stream < transporters_.size());

    UPTransporterBase& result = transporters_[stream.get()];
    if (!result)
    {
        result = [this, stream]() -> std::unique_ptr<TransporterBase> {
            // Thread-local transporter input
            TransporterInput local_trans_inp = *transporter_input_;
            local_trans_inp.stream_id = stream;

            if (use_device_)
            {
                CELER_VALIDATE(device(),
                               << "CUDA device is unavailable but GPU run was "
                                  "requested");
                return std::make_unique<Transporter<MemSpace::device>>(
                    std::move(local_trans_inp));
            }
            else
            {
                return std::make_unique<Transporter<MemSpace::host>>(
                    std::move(local_trans_inp));
            }
        }();
    }
    CELER_ENSURE(result);
    return *result;
}

//---------------------------------------------------------------------------//
/*!
 * Get an already-constructed transporter for the given stream.
 */
auto Runner::get_transporter_ptr(StreamId stream) const -> TransporterBase const*
{
    CELER_EXPECT(stream < transporters_.size());
    return transporters_[stream.get()].get();
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
