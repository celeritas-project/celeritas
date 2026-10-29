//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/StepperTestBase.cc
//---------------------------------------------------------------------------//
#include "StepperTestBase.hh"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <numeric>
#include <gtest/gtest.h>

#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"
#include "corecel/io/LogContextException.hh"
#include "corecel/io/Repr.hh"
#include "celeritas/global/ActionSequence.hh"
#include "celeritas/global/Stepper.hh"
#include "celeritas/phys/PhysicsParams.hh"

#include "AssertionHelper.hh"
#include "testdetail/TestMacrosImpl.hh"

using std::cout;

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
//! Generate a stepper construction class
StepperInput StepperTestBase::make_stepper_input(size_type tracks)
{
    CELER_EXPECT(tracks > 0);

    StepperInput result;
    result.params = this->core();
    result.stream_id = StreamId{0};
    result.num_track_slots = tracks;
    result.actions = std::make_shared<ActionSequence>(
        *this->action_reg(), ActionSequence::Options{});

    if (auto& d = device())
    {
        d.create_streams(1);
    }

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Get physics attributes for checking.
 */
auto StepperTestBase::check_setup() -> SetupCheckResult
{
    PhysicsParams const& p = *this->physics();
    SetupCheckResult result;

    for (auto process_id : range(ProcessId{p.num_processes()}))
    {
        result.processes.emplace_back(p.process(process_id)->label());
    }

    // Create temporary host stepper to get action ordering
    Stepper<MemSpace::host> temp_stepper(this->make_stepper_input(1));
    auto const& action_seq = temp_stepper.actions();
    for (auto const& sp_action : action_seq.actions().step())
    {
        result.actions.emplace_back(sp_action->label());
        result.actions_desc.emplace_back(sp_action->description());
    }

    return result;
}

//---------------------------------------------------------------------------//
auto StepperTestBase::run(StepperInterface& step,
                          size_type num_primaries) const -> RunResult
{
    // Perform first step
    auto primaries = this->make_primaries(num_primaries);

    StepperResult counts;
    CELER_TRY_HANDLE(counts = step(make_span(primaries)),
                     LogContextException{this->output_reg().get()});
    EXPECT_EQ(num_primaries, counts.active);
    EXPECT_GE(num_primaries, counts.alive);

    if (this->HasFailure())
    {
        return {};
    }

    RunResult result;
    result.active = {counts.active};
    result.queued = {counts.queued};

    size_type const max_steps = this->max_average_steps() * num_primaries;
    size_type accum_steps = counts.active;

    while (counts)
    {
        CELER_TRY_HANDLE(counts = step(),
                         LogContextException{this->output_reg().get()});
        result.active.push_back(counts.active);
        result.queued.push_back(counts.queued);
        accum_steps += counts.active;
        EXPECT_LT(accum_steps, max_steps) << "max steps exceeded";
        if (accum_steps >= max_steps)
        {
            break;
        }
    }
    return result;
}

//---------------------------------------------------------------------------//
// RUN RESULT
//---------------------------------------------------------------------------//
//! Cumulative number of steps over all tracks / number of primaries
double StepperTestBase::RunResult::calc_avg_steps_per_primary() const
{
    CELER_EXPECT(*this);
    using namespace celeritas::literals;
    size_type num_primaries = this->active.front();
    auto accum_steps
        = std::accumulate(this->active.begin(), this->active.end(), 0_sz);
    return static_cast<double>(accum_steps)
           / static_cast<double>(num_primaries);
}

//---------------------------------------------------------------------------//
/*!
 * Index of step after the final non-full capacity/maximum is reached.
 *
 * For example: \verbatim
    1, 2, 4, 8, 7, 8, 4, 3, 2, 1, 0
                      ^
   \endverbatim
 * returns 6.
 */
size_type StepperTestBase::RunResult::calc_emptying_step() const
{
    CELER_EXPECT(*this);
    auto iter = this->active.begin();
    // Iterator where previous value is high water mark and current is less
    auto result = iter;
    // Value of the previous step
    size_type prev = *iter++;
    // Unfortunately we don't know the capacity (and it might not even be
    // reached?) so find the highest value
    size_type max_cap = 0;
    while (iter != this->active.end())
    {
        max_cap = max(prev, max_cap);
        if (prev == max_cap && *iter < max_cap)
        {
            result = iter;
        }
        prev = *iter++;
    }
    return result - this->active.begin();
}

//---------------------------------------------------------------------------//
//! Index of the high water mark of the initializer queue
auto StepperTestBase::RunResult::calc_queue_hwm() const -> StepCount
{
    CELER_EXPECT(*this);
    auto iter = std::max_element(this->queued.begin(), this->queued.end());
    return {iter - this->queued.begin(), *iter};
}

//---------------------------------------------------------------------------//
std::ostream& operator<<(std::ostream& os,
                         StepperTestBase::SetupCheckResult const& ref)
{
    return os << '{' << repr(ref.processes) << ", " << repr(ref.actions)
              << ", " << repr(ref.actions_desc) << '}';
}

void check_ref_eq(AssertionHelper& helper,
                  StepperTestBase::SetupCheckResult const& expected,
                  StepperTestBase::SetupCheckResult const& actual,
                  std::nullptr_t)
{
    is_vec_eq(helper, "processes", expected.processes, actual.processes);
    is_vec_eq(helper, "actions", expected.actions, actual.actions);
    is_vec_eq(
        helper, "actions_desc", expected.actions_desc, actual.actions_desc);
}

//---------------------------------------------------------------------------//
std::ostream& operator<<(std::ostream& os,
                         StepperTestBase::RunResult const& ref)
{
    return os << '{' << repr(ref.active) << ", " << repr(ref.queued) << '}';
}

void check_ref_eq(AssertionHelper& helper,
                  StepperTestBase::RunResult const& expected,
                  StepperTestBase::RunResult const& actual,
                  std::nullptr_t)
{
    is_vec_eq(helper, "active", expected.active, actual.active);
    is_vec_eq(helper, "queued", expected.queued, actual.queued);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
