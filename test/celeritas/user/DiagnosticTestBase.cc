//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/DiagnosticTestBase.cc
//---------------------------------------------------------------------------//
#include "DiagnosticTestBase.hh"

#include <iostream>

#include "corecel/cont/Span.hh"
#include "corecel/io/OutputRegistry.hh"
#include "celeritas/global/ActionRegistry.hh"
#include "celeritas/global/Stepper.hh"
#include "celeritas/user/ActionDiagnostic.hh"

using std::cout;

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
DiagnosticTestBase::~DiagnosticTestBase() = default;

//---------------------------------------------------------------------------//
/*!
 * Construct action diagnostic at setup time.
 */
void DiagnosticTestBase::SetUp()
{
    size_type num_streams = 1;

    action_diagnostic_
        = std::make_shared<ActionDiagnostic>(this->action_reg()->next_id(),
                                             this->action_reg(),
                                             this->particle(),
                                             num_streams);

    // Add to action registry
    this->action_reg()->insert(action_diagnostic_);
    // Add to output interface
    this->output_reg()->insert(action_diagnostic_);
}

//---------------------------------------------------------------------------//
//! Print the expected result
void DiagnosticTestBase::RunResult::print_expected() const
{
    cout
        << "/*** ADD THE FOLLOWING UNIT TEST CODE ***/\n"
           "static const size_type expected_actions[] = "
        << repr(this->actions)
        << ";\n"
           "EXPECT_VEC_EQ(expected_actions, result.actions);\n"
           "static char const* expected_nonzero_actions[] = "
        << repr(this->nonzero_actions)
        << ";\n"
           "EXPECT_VEC_EQ(expected_nonzero_actions, result.nonzero_actions);\n"
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
    StepperInput step_inp;
    step_inp.params = this->core();
    step_inp.stream_id = StreamId{0};
    step_inp.num_track_slots = num_tracks;

    Stepper<M> step(step_inp);

    // Initial step
    auto primaries = this->make_primaries(num_tracks);
    auto count = step(make_span(primaries));

    while (count && --num_steps > 0)
    {
        count = step();
    }

    RunResult result;

    // Save action diagnostic results
    for (auto const& vec : action_diagnostic_->calc_actions())
    {
        result.actions.insert(result.actions.end(), vec.begin(), vec.end());
    }
    for (auto const& [label, _] : action_diagnostic_->calc_actions_map())
    {
        result.nonzero_actions.push_back(label);
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
template DiagnosticTestBase::RunResult
    DiagnosticTestBase::run<MemSpace::device>(size_type, size_type);
template DiagnosticTestBase::RunResult
    DiagnosticTestBase::run<MemSpace::host>(size_type, size_type);

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
