//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/StepperTestBase.cc
//---------------------------------------------------------------------------//
#include "StepperTestBase.hh"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <gtest/gtest.h>

#include "corecel/io/Repr.hh"
#include "celeritas/global/ActionManager.hh"
#include "celeritas/global/Stepper.hh"

using std::cout;

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
//! Construct dummy action at creation
StepperTestBase::StepperTestBase()
{
    auto& action_mgr = *this->action_mgr();

    static const char desc[] = "count the number of executions";
    dummy_action_            = std::make_shared<DummyAction>(
        action_mgr.next_id(), "dummy-action", desc);
    action_mgr.insert(dummy_action_);
}

//---------------------------------------------------------------------------//
//! Generate a stepper construction class
StepperInput
StepperTestBase::make_stepper_input(size_type tracks, size_type init_scaling)
{
    CELER_EXPECT(tracks > 0);
    CELER_EXPECT(init_scaling > 1);

    StepperInput result;
    result.params           = this->core();
    result.num_track_slots  = tracks;
    result.num_initializers = init_scaling * tracks;

    CELER_ASSERT(dummy_action_);
    result.post_step_callback = dummy_action_->action_id();
    CELER_ENSURE(result.post_step_callback);
    return result;
}

//---------------------------------------------------------------------------//
auto StepperTestBase::run(StepperInterface& step,
                          size_type         num_primaries) const -> RunResult
{
    CELER_EXPECT(step);

    // Perform first step
    auto counts = step(this->make_primaries(num_primaries));
    EXPECT_EQ(num_primaries, counts.active);
    EXPECT_EQ(num_primaries, counts.alive);

    RunResult result;
    result.active = {counts.active};
    result.queued = {counts.queued};

    const size_type max_steps   = this->max_average_steps() * num_primaries;
    size_type       accum_steps = counts.active;

    while (counts)
    {
        counts = step();
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
    size_type num_primaries = this->active.front();
    auto      accum_steps   = std::accumulate(
        this->active.begin(), this->active.end(), size_type{0});
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
//! Print the expected result
void StepperTestBase::RunResult::print_expected() const
{
    CELER_EXPECT(*this);
    auto step_value = this->calc_queue_hwm();
    cout << "/*** ADD THE FOLLOWING UNIT TEST CODE ***/\n"
            "EXPECT_EQ("
         << this->num_step_iters()
         << ", result.num_step_iters());\n"
            "EXPECT_SOFT_EQ("
         << repr(this->calc_avg_steps_per_primary())
         << ", result.calc_avg_steps_per_primary());\n"
            "EXPECT_EQ("
         << this->calc_emptying_step()
         << ", result.calc_emptying_step());\n"
            "EXPECT_EQ(RunResult::StepCount({"
         << step_value.first << ", " << step_value.second
         << "}), result.calc_queue_hwm());\n"
            "/*** END CODE ***/\n";
}
//---------------------------------------------------------------------------//
} // namespace test
} // namespace celeritas
