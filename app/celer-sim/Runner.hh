//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-sim/Runner.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <unordered_map>
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
class ParticleParams;
class RootFileManager;
class StepCollector;
}  // namespace celeritas

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
struct RunnerInput;

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
    using Input = RunnerInput;
    using MapStrDouble = std::unordered_map<std::string, double>;
    using RunnerResult = TransporterResult;
    using SPOutputRegistry = std::shared_ptr<OutputRegistry>;
    //!@}

  public:
    // Construct on all threads from a JSON input and shared output manager
    Runner(RunnerInput const& inp, SPOutputRegistry output);

    // Warm up by running a single step with no active tracks
    void warm_up();

    // Run on a single stream/thread, returning the transport result
    RunnerResult operator()(StreamId, EventId);

    // Run all events simultaneously on a single stream
    RunnerResult operator()();

    // Number of streams supported
    StreamId::size_type num_streams() const;

    // Total number of events
    size_type num_events() const;

    // Get the accumulated action times
    MapStrDouble get_action_times() const;

  private:
    //// TYPES ////

    using UPTransporterBase = std::unique_ptr<TransporterBase>;
    using SPConstParticles = std::shared_ptr<ParticleParams const>;
    using VecPrimary = std::vector<Primary>;
    using VecEvent = std::vector<VecPrimary>;

    //// DATA ////

    std::shared_ptr<CoreParams> core_params_;
    std::shared_ptr<RootFileManager> root_manager_;
    std::shared_ptr<StepCollector> step_collector_;

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
    size_type build_events(RunnerInput const&, SPConstParticles);
    TransporterBase& get_transporter(StreamId);
    TransporterBase const* get_transporter_ptr(StreamId) const;
};

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
