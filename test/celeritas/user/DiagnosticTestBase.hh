//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/DiagnosticTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>

#include "corecel/io/Label.hh"
#include "corecel/io/Repr.hh"
#include "celeritas/phys/Primary.hh"

#include "StepCollectorTestBase.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class ActionDiagnostic;
class StepDiagnostic;

namespace test
{
//---------------------------------------------------------------------------//
class DiagnosticTestBase : virtual public StepCollectorTestBase
{
  public:
    //!@{
    //! \name Type aliases
    using VecPrimary = std::vector<Primary>;
    //!@}

    struct RunResult
    {
        std::vector<std::string> nonzero_action_keys;
        std::vector<size_type> nonzero_action_counts;
        std::vector<size_type> steps;

        void print_expected() const;
    };

  public:
    // Default destructor
    ~DiagnosticTestBase();

    void SetUp() override;

    template<MemSpace M>
    RunResult run(size_type num_tracks, size_type num_steps);

    // Get JSON output from the action diagnostic
    std::string action_output() const;

    // Get JSON output from the step diagnostic
    std::string step_output() const;

    // Print expected results
    void print_expected() const;

  protected:
    std::shared_ptr<ActionDiagnostic> action_diagnostic_;
    std::shared_ptr<StepDiagnostic> step_diagnostic_;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
