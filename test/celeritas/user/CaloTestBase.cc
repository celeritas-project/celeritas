//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/CaloTestBase.cc
//---------------------------------------------------------------------------//
#include "CaloTestBase.hh"

#include <iostream>

#include "corecel/cont/Span.hh"
#include "celeritas/global/Stepper.hh"
#include "celeritas/user/StepCollector.hh"

#include "ExampleCalorimeters.hh"

using std::cout;

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
CaloTestBase::~CaloTestBase() = default;

//---------------------------------------------------------------------------//
/*!
 * Construct example callback and step collector at setup time.
 */
void CaloTestBase::SetUp()
{
    example_calos_ = std::make_shared<ExampleCalorimeters>(
        *this->geometry(), this->get_detector_names());

    StepCollector::VecInterface interfaces = {example_calos_};

    collector_ = std::make_shared<StepCollector>(
        std::move(interfaces), this->geometry(), this->action_reg().get());
}

//---------------------------------------------------------------------------//
//! Print the expected result
void CaloTestBase::RunResult::print_expected() const
{
    cout << "/*** ADD THE FOLLOWING UNIT TEST CODE ***/\n"
            "static const double expected_edep[] = "
         << repr(this->edep)
         << ";\n"
            "EXPECT_VEC_SOFT_EQ(expected_edep, result.edep);\n"
            "/*** END CODE ***/\n";
}

//---------------------------------------------------------------------------//
/*!
 * Run a number of tracks.
 */
auto CaloTestBase::run(size_type num_tracks, size_type num_steps) -> RunResult
{
    StepperInput step_inp;
    step_inp.params = this->core();
    step_inp.num_track_slots = num_tracks;

    Stepper<MemSpace::host> step(step_inp);

    // Initial step
    auto primaries = this->make_primaries(num_tracks);
    auto count = step(make_span(primaries));

    while (count && --num_steps > 0)
    {
        count = step();
    }

    RunResult result;
    auto edep = example_calos_->deposition();
    result.edep.assign(edep.begin(), edep.end());
    example_calos_->clear();

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
