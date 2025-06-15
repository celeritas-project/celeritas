//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/DiagnosticTestBase.cc
//---------------------------------------------------------------------------//
#include "DiagnosticTestBase.hh"

#include <iostream>

#include "corecel/cont/Span.hh"
#include "corecel/io/OutputRegistry.hh"
#include "corecel/sys/ActionRegistry.hh"
#include "celeritas/global/Stepper.hh"
#include "celeritas/user/ActionDiagnostic.hh"
#include "celeritas/user/StepDiagnostic.hh"

using std::cout;

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
DiagnosticTestBase::~DiagnosticTestBase() = default;

//---------------------------------------------------------------------------//
/*!
 * Construct diagnostics at setup time.
 */
void DiagnosticTestBase::SetUp()
{
    // Create action diagnostic
    action_diagnostic_ = ActionDiagnostic::make_and_insert(*this->core());

    // Create step diagnostic
    step_diagnostic_ = StepDiagnostic::make_and_insert(*this->core(), 20);
}

//---------------------------------------------------------------------------//
//! Print the expected result
void DiagnosticTestBase::RunResult::print_expected() const
{
    cout << "/*** ADD THE FOLLOWING UNIT TEST CODE ***/\n"
            "static char const* const expected_nonzero_action_keys[] = "
         << repr(this->nonzero_action_keys)
         << ";\n"
            "EXPECT_VEC_EQ(expected_nonzero_action_keys, "
            "result.nonzero_action_keys);\n"
            "static size_type const expected_nonzero_action_counts[] = "
         << repr(this->nonzero_action_counts)
         << ";\n"
            "EXPECT_VEC_EQ(expected_nonzero_action_counts, "
            "result.nonzero_action_counts);\n"
            "static size_type const expected_steps[] = "
         << repr(this->steps)
         << ";\n"
            "EXPECT_VEC_EQ(expected_steps, result.steps);\n"
            "/*** END CODE ***/\n";
}

//---------------------------------------------------------------------------//
//! Print the expected result
void DiagnosticTestBase::print_expected() const
{
    cout << "/*** ADD THE FOLLOWING UNIT TEST CODE ***/\n"
         << "EXPECT_EQ(R\"json(" << this->action_output()
         << ")json\",this->action_output());\n"
         << "EXPECT_EQ(R\"json(" << this->step_output()
         << ")json\",this->step_output());\n"
            "/*** END CODE ***/\n";
}

//---------------------------------------------------------------------------//
/*!
 * Run a number of tracks.
 */
template<MemSpace M>
auto DiagnosticTestBase::run(size_type num_tracks, size_type num_steps)
    -> RunResult
{
    this->run_impl<M>(num_tracks, num_steps);

    RunResult result;

    // Save action diagnostic results
    for (auto const& [label, count] : action_diagnostic_->calc_actions_map())
    {
        result.nonzero_action_keys.push_back(label);
        result.nonzero_action_counts.push_back(count);
    }

    // Save step diagnostic results
    for (auto const& vec : step_diagnostic_->calc_steps())
    {
        result.steps.insert(result.steps.end(), vec.begin(), vec.end());
    }

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Get output from the action diagnostic.
 */
std::string DiagnosticTestBase::action_output() const
{
    // See OutputInterface.hh
    return to_string(*action_diagnostic_);
}

//---------------------------------------------------------------------------//
/*!
 * Get output from the step diagnostic.
 */
std::string DiagnosticTestBase::step_output() const
{
    // See OutputInterface.hh
    return to_string(*step_diagnostic_);
}

//---------------------------------------------------------------------------//
template DiagnosticTestBase::RunResult
    DiagnosticTestBase::run<MemSpace::device>(size_type, size_type);
template DiagnosticTestBase::RunResult
    DiagnosticTestBase::run<MemSpace::host>(size_type, size_type);

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
