//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-loop/Runner.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "corecel/Types.hh"
#include "corecel/sys/ThreadId.hh"
#include "celeritas/phys/Primary.hh"

#include "Transporter.hh"

namespace celeritas
{
class CoreParams;
class OutputRegistry;
class RootFileManager;
class StepCollector;
}  // namespace celeritas

namespace demo_loop
{
//---------------------------------------------------------------------------//
struct RunnerInput;

//---------------------------------------------------------------------------//
/*!
 * Results from transporting all events.
 */
struct SimulationResult
{
    //!@{
    //! \name Type aliases
    using real_type = celeritas::real_type;
    //!@}

    //// DATA ////

    real_type total_time{};  //!< Total simulation time
    real_type setup_time{};  //!< One-time initialization cost
    std::vector<TransporterResult> events;  //< Results tallied for each event
};

//---------------------------------------------------------------------------//
/*!
 * Manage execution of Celeritas.
 *
 * This class is meant to be created in a single-thread context, and executed
 * in a multi-thread context.
 */
class Runner
{
  public:
    //!@{
    //! \name Type aliases
    using EventId = celeritas::EventId;
    using StreamId = celeritas::StreamId;
    using size_type = celeritas::size_type;
    using Input = RunnerInput;
    using RunnerResult = TransporterResult;
    using SPOutputRegistry = std::shared_ptr<celeritas::OutputRegistry>;
    using UPTransporterBase = std::unique_ptr<TransporterBase>;
    //!@}

    //! ID of the stream and event to be run
    struct RunStreamEvent
    {
        StreamId stream{};
        EventId event{};
    };

  public:
    // Construct on all threads from a JSON input and shared output manager
    Runner(RunnerInput const& inp, SPOutputRegistry output);

    // Run on a single stream/thread, returning the transport result
    RunnerResult operator()(RunStreamEvent);

    // Number of streams supported
    StreamId::size_type num_streams() const;

    // Total number of events
    size_type num_events() const;

  private:
    //// TYPES ////

    using VecEvent = std::vector<std::vector<celeritas::Primary>>;

    //// DATA ////

    std::shared_ptr<celeritas::CoreParams> core_params_;
    std::shared_ptr<celeritas::RootFileManager> root_manager_;
    std::shared_ptr<celeritas::StepCollector> step_collector_;

    // Transporter inputs and stream-local transporters
    bool use_device_{};
    std::shared_ptr<TransporterInput> transporter_input_;
    VecEvent events_;
    std::vector<UPTransporterBase> transporters_;

    //// HELPER FUNCTIONS ////

    void setup_globals(RunnerInput const&) const;
    void build_core_params(RunnerInput const&, SPOutputRegistry&&);
    void build_step_collectors(RunnerInput const&);
    void build_diagnostics(RunnerInput const&);
    void build_transporter_input(RunnerInput const&);
    void build_events(RunnerInput const&);
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
// Get the number of streams from the OMP_NUM_THREADS environment variable
int get_num_streams();

//---------------------------------------------------------------------------//
}  // namespace demo_loop
