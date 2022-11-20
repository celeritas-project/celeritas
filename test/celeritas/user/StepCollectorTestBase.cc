//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/StepCollectorTestBase.cc
//---------------------------------------------------------------------------//
#include "StepCollectorTestBase.hh"

#include <iostream>

#include "celeritas/global/Stepper.hh"
#include "celeritas/user/StepCollector.hh"

#include "ExampleMctruth.hh"

using std::cout;

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
StepCollectorTestBase::~StepCollectorTestBase() = default;

//---------------------------------------------------------------------------//
/*!
 * Construct example callback and step collector at setup time.
 */
void StepCollectorTestBase::SetUp()
{
    example_mctruth_ = std::make_shared<ExampleMctruth>();

    StepCollector::VecInterface interfaces = {example_mctruth_};

    collector_ = std::make_shared<StepCollector>(std::move(interfaces),
                                                 this->action_reg().get());
}

//---------------------------------------------------------------------------//
//! Print the expected result
void StepCollectorTestBase::RunResult::print_expected() const
{
    cout << "/*** ADD THE FOLLOWING UNIT TEST CODE ***/\n"
            "static const int expected_event[] = "
         << repr(this->event)
         << ";\n"
            "EXPECT_VEC_EQ(expected_event, result.event);\n"
            "static const int expected_track[] = "
         << repr(this->track)
         << ";\n"
            "EXPECT_VEC_EQ(expected_track, result.track);\n"
            "static const int expected_step[] = "
         << repr(this->step)
         << ";\n"
            "EXPECT_VEC_EQ(expected_step, result.step);\n"
            "static const int expected_volume[] = "
         << repr(this->volume)
         << ";\n"
            "EXPECT_VEC_EQ(expected_volume, result.volume);\n"
            "static const double expected_pos[] = "
         << repr(this->pos)
         << ";\n"
            "EXPECT_VEC_SOFT_EQ(expected_pos, result.pos);\n"
            "static const double expected_dir[] = "
         << repr(this->dir)
         << ";\n"
            "EXPECT_VEC_SOFT_EQ(expected_dir, result.dir);\n"
            "/*** END CODE ***/\n";
}

//---------------------------------------------------------------------------//
/*!
 * Run a number of tracks.
 */
auto StepCollectorTestBase::run(size_type num_tracks, size_type num_steps)
    -> RunResult
{
    StepperInput step_inp;
    step_inp.params           = this->core();
    step_inp.num_track_slots  = num_tracks;

    Stepper<MemSpace::host> step(step_inp);

    // Initial step
    auto count = step(this->make_primaries(num_tracks));

    while (count && --num_steps > 0)
    {
        count = step();
    }

    example_mctruth_->sort();

    RunResult result;
    for (const ExampleMctruth::Step& s : example_mctruth_->steps())
    {
        result.event.push_back(s.event);
        result.track.push_back(s.track);
        result.step.push_back(s.step);
        result.volume.push_back(s.volume);
        result.pos.insert(result.pos.end(), std::begin(s.pos), std::end(s.pos));
        result.dir.insert(result.dir.end(), std::begin(s.dir), std::end(s.dir));
    }
    return result;
}

//---------------------------------------------------------------------------//
} // namespace test
} // namespace celeritas
