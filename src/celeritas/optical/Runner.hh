//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/Runner.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "celeritas/inp/StandaloneInput.hh"
#include "celeritas/setup/StandaloneInput.hh"
#include "celeritas/user/ActionTimes.hh"
#include "celeritas/user/StepTimes.hh"

#include "Transporter.hh"
#include "gen/DirectGeneratorAction.hh"
#include "gen/GeneratorAction.hh"
#include "gen/PrimaryGeneratorAction.hh"

namespace celeritas
{
namespace optical
{
class CoreStateBase;
class CoreParams;

//---------------------------------------------------------------------------//
/*!
 * Manage execution of a standalone Celeritas optical stepping loop.
 *
 * \note When parallelizing on the CPU using OpenMP, this class expects
 * track-level parallelism be enabled (transporting all tracks with a single
 * stream and state) rather than event-level parallelism (transporting events
 * on separate streams with one state per stream). Similarly on the GPU a
 * single stream and state will be used.
 */
class Runner
{
  public:
    //!@{
    //! \name Type aliases
    using Input = inp::OpticalStandaloneInput;
    using SPConstParams = std::shared_ptr<CoreParams const>;
    using SpanConstTrackInit = DirectGeneratorAction::SpanConstData;
    using SpanConstGenDist = GeneratorAction::SpanConstData;
    //!@}

    struct Result
    {
        CounterAccumStats counters;
        ActionTimes::MapStrDbl action_times;
        StepTimes::VecDbl step_times;
    };

  public:
    // Construct with optical problem input definition
    explicit Runner(Input&&);

    // Set the number of pending tracks for a primary generator
    void insert();

    // Insert track initializers
    void insert(SpanConstTrackInit);

    // Insert distributions for generating through scintillation or Cherenkov
    void insert(SpanConstGenDist);

    // Generate optical photons and transport to completion
    Result operator()() const;

    //! Access the shared params
    SPConstParams const& params() const
    {
        return loaded_.problem.transporter->params();
    }

    //! Access the problem setup
    setup::OpticalProblemLoaded const& problem() const
    {
        return loaded_.problem;
    }

  private:
    setup::OpticalStandaloneLoaded loaded_;
    std::shared_ptr<CoreStateBase> state_;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
