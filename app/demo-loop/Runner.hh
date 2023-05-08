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
 * Simulation timing results.
 *
 * TODO: maybe a timer diagnostic class could help out here?
 * or another OutputRegistry.
 */
struct RunTimingResult
{
    using real_type = celeritas::real_type;
    using VecReal = std::vector<real_type>;
    using MapStrReal = std::unordered_map<std::string, real_type>;

    VecReal steps;  //!< Real time per step
    real_type total{};  //!< Total simulation time
    real_type setup{};  //!< One-time initialization cost
    MapStrReal actions{};  //!< Accumulated action timing
};

//---------------------------------------------------------------------------//
/*!
 * Tallied result and timing from transporting a set of primaries.
 *
 * TODO: these should be migrated to OutputInterface classes.
 */
struct RunnerResult
{
    //!@{
    //! \name Type aliases
    using size_type = celeritas::size_type;
    using VecCount = std::vector<size_type>;
    //!@}

    //// DATA ////

    VecCount initializers;  //!< Num starting track initializers
    VecCount active;  //!< Num tracks active at beginning of step
    VecCount alive;  //!< Num living tracks at end of step
    RunTimingResult time;  //!< Timing information
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
    using StreamId = celeritas::StreamId;
    using Input = RunnerInput;
    using SPOutputRegistry = std::shared_ptr<celeritas::OutputRegistry>;
    //!@}

  public:
    // Construct on all threads from a JSON input and shared output manager
    Runner(RunnerInput const& inp, SPOutputRegistry output);

    // Run on a single stream/thread, returning the transport result
    RunnerResult operator()(StreamId s) const;

    // Number of streams supported
    StreamId::size_type num_streams() const;

  private:
    //// DATA ////

    std::shared_ptr<celeritas::CoreParams> core_params_;
    std::shared_ptr<celeritas::RootFileManager> root_manager_;
    std::shared_ptr<celeritas::StepCollector> step_collector_;

    // Transporter inputs
    bool use_device_{};
    std::shared_ptr<TransporterInput> transporter_input_;
    std::vector<celeritas::Primary> primaries_;

    //// HELPER FUNCTIONS ////

    void setup_globals(RunnerInput const&) const;
    void build_core_params(RunnerInput const&, SPOutputRegistry&&);
    void build_step_collectors(RunnerInput const&);
    void build_diagnostics(RunnerInput const&);
    void build_transporter_input(RunnerInput const&);
    void build_primaries(RunnerInput const&);
};

//---------------------------------------------------------------------------//
}  // namespace demo_loop
