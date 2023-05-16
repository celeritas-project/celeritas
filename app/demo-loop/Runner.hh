//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-loop/Runner.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "corecel/Types.hh"
#include "corecel/sys/ThreadId.hh"
#include "celeritas/phys/Primary.hh"

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
struct TransporterInput;

//---------------------------------------------------------------------------//
/*!
 * Tallied result and timing from transporting a single event.
 */
struct RunnerResult
{
    using real_type = celeritas::real_type;
    using size_type = celeritas::size_type;
    using MapStrReal = std::unordered_map<std::string, real_type>;
    using VecReal = std::vector<real_type>;
    using VecCount = std::vector<size_type>;

    VecCount initializers;  //!< Num starting track initializers
    VecCount active;  //!< Num tracks active at beginning of step
    VecCount alive;  //!< Num living tracks at end of step
    MapStrReal action_times{};  //!< Accumulated action timing
    VecReal step_times;  //!< Real time per step
};

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
    std::vector<RunnerResult> events;  //< Results tallied for each event
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
    using SPOutputRegistry = std::shared_ptr<celeritas::OutputRegistry>;
    //!@}

  public:
    // Construct on all threads from a JSON input and shared output manager
    Runner(RunnerInput const& inp, SPOutputRegistry output);

    // Run on a single stream/thread, returning the transport result
    RunnerResult operator()(StreamId, EventId) const;

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

    // Transporter inputs
    bool use_device_{};
    std::shared_ptr<TransporterInput> transporter_input_;
    VecEvent events_;

    //// HELPER FUNCTIONS ////

    void setup_globals(RunnerInput const&) const;
    void build_core_params(RunnerInput const&, SPOutputRegistry&&);
    void build_step_collectors(RunnerInput const&);
    void build_diagnostics(RunnerInput const&);
    void build_transporter_input(RunnerInput const&);
    void build_events(RunnerInput const&);
};

//---------------------------------------------------------------------------//
}  // namespace demo_loop
