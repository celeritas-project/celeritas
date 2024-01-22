//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-sim/RunnerOutput.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/io/OutputInterface.hh"

#include "Transporter.hh"

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Results from transporting all events.
 */
struct SimulationResult
{
    using MapStrDouble = std::unordered_map<std::string, double>;

    //// DATA ////

    double total_time{};  //!< Total simulation time
    double setup_time{};  //!< One-time initialization cost
    double warmup_time{};  //!< One-time warmup cost
    MapStrDouble action_times{};  //!< Accumulated mean action wall times
    std::vector<TransporterResult> events;  //!< Results tallied for each event
    size_type num_streams{};  //!< Number of CPU/OpenMP threads
};

//---------------------------------------------------------------------------//
/*!
 * Output demo loop results.
 */
class RunnerOutput final : public OutputInterface
{
  public:
    // Construct from simulation result
    explicit RunnerOutput(SimulationResult result);

    //! Category of data to write
    Category category() const final { return Category::result; }

    //! Name of the entry inside the category.
    std::string label() const final { return "runner"; }

    // Write output to the given JSON object
    void output(JsonPimpl*) const final;

  private:
    SimulationResult result_;
};

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
