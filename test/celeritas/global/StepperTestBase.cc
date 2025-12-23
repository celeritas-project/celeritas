//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/StepperTestBase.cc
//---------------------------------------------------------------------------//
#include "StepperTestBase.hh"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <gtest/gtest.h>

#include "corecel/cont/Span.hh"
#include "corecel/io/LogContextException.hh"
#include "corecel/io/Repr.hh"
#include "celeritas/global/ActionSequence.hh"
#include "celeritas/global/Stepper.hh"
#include "celeritas/phys/PhysicsParams.hh"

using std::cout;

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
//! Whether the build uses the default real type and RNG.
bool StepperTestBase::is_default_build()
{
    return CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE
           && CELERITAS_UNITS == CELERITAS_UNITS_CGS
           && CELERITAS_CORE_RNG == CELERITAS_CORE_RNG_XORWOW;
}

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
    size_type num_primaries = this->active.front();
    auto accum_steps = std::accumulate(
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
void StepperTestBase::SetupCheckResult::print_expected() const
{
    cout << "/*** ADD THE FOLLOWING UNIT TEST CODE ***/\n"
            "static const char* const expected_processes[] = "
         << repr(this->processes)
         << ";\n"
            "EXPECT_VEC_EQ(expected_processes, result.processes);\n"
            "static const char* const expected_actions[] = "
         << repr(this->actions)
         << ";\n"
            "EXPECT_VEC_EQ(expected_actions, result.actions);\n"
            "static const char* const expected_actions_desc[] = "
         << repr(this->actions_desc)
         << ";\n"
            "EXPECT_VEC_EQ(expected_actions_desc, result.actions_desc);\n"
            "/*** END CODE ***/\n";
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
}  // namespace test
}  // namespace celeritas
