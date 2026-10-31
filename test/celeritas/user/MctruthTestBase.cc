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
#include "RefComparer.hh"
#include "TestMacros.hh"

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
std::ostream& operator<<(std::ostream& os,
                         MctruthTestBase::RunResult const& ref)
{
    return os << '{' << CELER_REPR_ATTR(ref, event) << ", "
              << CELER_REPR_ATTR(ref, track) << ", "
              << CELER_REPR_ATTR(ref, step) << ", "
              << CELER_REPR_ATTR(ref, volume) << ", "
              << CELER_REPR_ATTR(ref, pos) << ", " << CELER_REPR_ATTR(ref, dir)
              << '}';
}

bool is_ref_eq(AssertionHelper& helper,
               MctruthTestBase::RunResult const& expected,
               MctruthTestBase::RunResult const& actual,
               MctruthTestBase::RunResult::Tolerance const& tol)
{
    using Result = MctruthTestBase::RunResult;
    RefComparer<Result> compare{helper, expected, actual};
    CELER_INVOKE_ATTR(compare.is_eq, Result, event);
    CELER_INVOKE_ATTR(compare.is_eq, Result, track);
    CELER_INVOKE_ATTR(compare.is_eq, Result, step);
    CELER_INVOKE_ATTR(compare.is_eq, Result, volume);
    CELER_INVOKE_ATTR(compare.is_soft_eq, Result, pos, tol.pos);
    CELER_INVOKE_ATTR(compare.is_soft_eq, Result, dir, tol.dir);
    return static_cast<bool>(compare);
}

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
        result.pos.push_back(
            {to_cm(s.pos[0]), to_cm(s.pos[1]), to_cm(s.pos[2])});
        result.dir.push_back({s.dir[0], s.dir[1], s.dir[2]});
    }
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
