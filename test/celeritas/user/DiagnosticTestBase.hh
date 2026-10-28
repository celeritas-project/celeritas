//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/DiagnosticTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>
#include <iosfwd>
#include <memory>
#include <vector>

#include "corecel/io/Label.hh"
#include "corecel/io/Repr.hh"
#include "celeritas/phys/Primary.hh"

#include "SimpleLoopTestBase.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class ActionDiagnostic;
class StepDiagnostic;

namespace test
{
class AssertionHelper;

//---------------------------------------------------------------------------//
class DiagnosticTestBase : virtual public SimpleLoopTestBase
{
  public:
    //!@{
    //! \name Type aliases
    using VecPrimary = std::vector<Primary>;
    //!@}

    struct RunResult
    {
        static constexpr char class_str[] = "RunResult";

        std::vector<std::string> nonzero_action_keys;
        std::vector<size_type> nonzero_action_counts;
        std::vector<size_type> steps;

        void print_expected() const;

        friend std::ostream& operator<<(std::ostream&, RunResult const&);
        friend void check_ref_eq(AssertionHelper&,
                                 RunResult const&,
                                 RunResult const&,
                                 std::nullptr_t);
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
