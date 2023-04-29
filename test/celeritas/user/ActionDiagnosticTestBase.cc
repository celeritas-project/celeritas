//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/ActionDiagnosticTestBase.cc
//---------------------------------------------------------------------------//
#include "ActionDiagnosticTestBase.hh"

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
ActionDiagnosticTestBase::~ActionDiagnosticTestBase() = default;

//---------------------------------------------------------------------------//
/*!
 * Construct action diagnostic at setup time.
 */
void ActionDiagnosticTestBase::SetUp()
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
void ActionDiagnosticTestBase::RunResult::print_expected() const
{
    cout << "/*** ADD THE FOLLOWING UNIT TEST CODE ***/\n"
            "static const size_type expected_counts[] = "
         << repr(this->counts)
         << ";\n"
            "EXPECT_VEC_EQ(expected_counts, result.counts);\n"
            "/*** END CODE ***/\n";
}

//---------------------------------------------------------------------------//
/*!
 * Run a number of tracks.
 */
template<MemSpace M>
auto ActionDiagnosticTestBase::run(size_type num_tracks, size_type num_steps)
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
    result.counts = action_diagnostic_->calc_particle_actions();

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Get output from the diagnostic.
 */
std::string ActionDiagnosticTestBase::output() const
{
    // See OutputInterface.hh
    return to_string(*action_diagnostic_);
}

//---------------------------------------------------------------------------//
template ActionDiagnosticTestBase::RunResult
    ActionDiagnosticTestBase::run<MemSpace::device>(size_type, size_type);
template ActionDiagnosticTestBase::RunResult
    ActionDiagnosticTestBase::run<MemSpace::host>(size_type, size_type);

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
