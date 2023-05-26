//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-loop/RunnerOutput.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/io/OutputInterface.hh"

#include "Runner.hh"

namespace demo_loop
{
//---------------------------------------------------------------------------//
/*!
 * Output demo loop results.
 */
class RunnerOutput final : public celeritas::OutputInterface
{
  public:
    //!@{
    //! \name Type aliases
    using JsonPimpl = celeritas::JsonPimpl;
    //!@}

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
}  // namespace demo_loop
