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

#include "Transporter.hh"

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Construct on all threads from a JSON input and shared output manager.
 */
Runner::Runner(Input si)
{
    // Set up problem
    auto loaded = setup::standalone_input(si);
    core_params_ = std::move(loaded.problem.core_params);
    CELER_ASSERT(core_params_);
    events_ = std::move(loaded.events);
    use_device_ = static_cast<bool>(si.system.device);

    CELER_VALIDATE(si.problem.tracking.limits.step_iters > 0,
                   << "nonpositive max step_iters="
                   << si.problem.tracking.limits.step_iters);

    transporter_input_ = std::make_shared<TransporterInput>();
    transporter_input_->optical = std::move(loaded.problem.optical_collector);
    transporter_input_->params = core_params_;
    transporter_input_->max_steps = si.problem.tracking.limits.step_iters;
    transporter_input_->store_track_counts
        = si.problem.diagnostics.counters.step;
    transporter_input_->actions = std::move(loaded.problem.actions);
    transporter_input_->log_progress = si.problem.diagnostics.log_frequency;

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
    return core_params_->sizes().streams;
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
 * Get the step times for each stream.
 */
auto Runner::get_step_times() const -> VecVecDouble
{
    VecVecDouble result(this->num_streams());
    for (auto sid : range(StreamId{this->num_streams()}))
    {
        if (auto* transport = this->get_transporter_ptr(sid))
        {
            result[sid.get()] = transport->get_step_times();
        }
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
auto Runner::get_transporter_ptr(StreamId stream) const
    -> TransporterBase const*
{
    CELER_EXPECT(stream < transporters_.size());
    return transporters_[stream.get()].get();
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
