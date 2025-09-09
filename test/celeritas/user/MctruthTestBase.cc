//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/MctruthTestBase.cc
//---------------------------------------------------------------------------//
#include "MctruthTestBase.hh"

#include <iostream>

#include "corecel/cont/Span.hh"
#include "corecel/io/LogContextException.hh"
#include "geocel/UnitUtils.hh"
#include "celeritas/global/Stepper.hh"
#include "celeritas/user/StepCollector.hh"

#include "ExampleMctruth.hh"

using std::cout;

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
MctruthTestBase::~MctruthTestBase() = default;

//---------------------------------------------------------------------------//
/*!
 * Construct example callback and step collector at setup time.
 */
void MctruthTestBase::SetUp()
{
    example_mctruth_ = std::make_shared<ExampleMctruth>();

    StepCollector::VecInterface interfaces = {example_mctruth_};

    collector_
        = StepCollector::make_and_insert(*this->core(), std::move(interfaces));
}

//---------------------------------------------------------------------------//
//! Print the expected result
void MctruthTestBase::RunResult::print_expected() const
{
    cout << "/*** ADD THE FOLLOWING UNIT TEST CODE ***/\n"
            "static int const expected_event[] = "
         << repr(this->event)
         << ";\n"
            "EXPECT_VEC_EQ(expected_event, result.event);\n"
            "static int const expected_track[] = "
         << repr(this->track)
         << ";\n"
            "EXPECT_VEC_EQ(expected_track, result.track);\n"
            "static int const expected_step[] = "
         << repr(this->step)
         << ";\n"
            "EXPECT_VEC_EQ(expected_step, result.step);\n"
            "static int const expected_volume[] = "
         << repr(this->volume)
         << ";\n"
            "EXPECT_VEC_EQ(expected_volume, result.volume);\n"
            "static double const expected_pos[] = "
         << repr(this->pos)
         << ";\n"
            "EXPECT_VEC_SOFT_EQ(expected_pos, result.pos);\n"
            "static double const expected_dir[] = "
         << repr(this->dir)
         << ";\n"
            "EXPECT_VEC_SOFT_EQ(expected_dir, result.dir);\n"
            "/*** END CODE ***/\n";
}

//---------------------------------------------------------------------------//
/*!
 * Run a number of tracks.
 */
auto MctruthTestBase::run(size_type num_tracks, size_type num_steps)
    -> RunResult
{
    this->run_impl<MemSpace::host>(num_tracks, num_steps);

    example_mctruth_->sort();

    RunResult result;
    for (ExampleMctruth::Step const& s : example_mctruth_->steps())
    {
        result.event.push_back(s.event);
        result.track.push_back(s.track);
        result.step.push_back(s.step);  // Step *counter* not *length*
        result.volume.push_back(s.volume);
        for (auto pos_v : s.pos)
        {
            result.pos.push_back(to_cm(pos_v));
        }
        result.dir.insert(result.dir.end(), std::begin(s.dir), std::end(s.dir));
    }
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
