//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/DormandPrinceStepper.cc
//---------------------------------------------------------------------------//
#include "DormandPrinceStepper.test.hh"

#include "celeritas_test.hh"  // for CELER_EXPECT and Test

namespace celeritas
{
namespace test
{

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class DormandPrinceStepperTest : public Test
{
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
TEST_F(DormandPrinceStepperTest, result_global_memory)
{
    auto expected = simulate_multi_next_chord(one_thread, number_states_sample);
    auto actual = simulate_multi_next_chord(multi_thread, number_states_sample);

    for (int i = 0; i < number_states_sample; i++)
    {
        EXPECT_VEC_SOFT_EQ(expected.results[i].mid_state.pos,
                           actual.results[i].mid_state.pos);
        EXPECT_VEC_SOFT_EQ(expected.results[i].mid_state.mom,
                           actual.results[i].mid_state.mom);

        EXPECT_VEC_SOFT_EQ(expected.results[i].end_state.pos,
                           actual.results[i].end_state.pos);
        EXPECT_VEC_SOFT_EQ(expected.results[i].end_state.mom,
                           actual.results[i].end_state.mom);

        EXPECT_VEC_SOFT_EQ(expected.results[i].err_state.pos,
                           actual.results[i].err_state.pos);
        EXPECT_VEC_SOFT_EQ(expected.results[i].err_state.mom,
                           actual.results[i].err_state.mom);
    }
}

TEST_F(DormandPrinceStepperTest, time_global_memory)
{
    auto old = simulate_multi_next_chord(one_thread, number_states_sample);
    auto multi_threaded
        = simulate_multi_next_chord(multi_thread, number_states_sample);

    CELER_VALIDATE(multi_threaded.milliseconds <= old.milliseconds,
                   << "Multi-threaded (" << multi_threaded.milliseconds
                   << " ms) is slower than old implementation ("
                   << old.milliseconds << " ms)");
}

TEST_F(DormandPrinceStepperTest, result_shared_memory)
{
    auto expected = simulate_multi_next_chord(one_thread, number_states_sample);
    auto actual
        = simulate_multi_next_chord(multi_thread, number_states_sample, true);

    for (int i = 0; i < number_states_sample; i++)
    {
        EXPECT_VEC_SOFT_EQ(expected.results[i].mid_state.pos,
                           actual.results[i].mid_state.pos);
        EXPECT_VEC_SOFT_EQ(expected.results[i].mid_state.mom,
                           actual.results[i].mid_state.mom);

        EXPECT_VEC_SOFT_EQ(expected.results[i].end_state.pos,
                           actual.results[i].end_state.pos);
        EXPECT_VEC_SOFT_EQ(expected.results[i].end_state.mom,
                           actual.results[i].end_state.mom);

        EXPECT_VEC_SOFT_EQ(expected.results[i].err_state.pos,
                           actual.results[i].err_state.pos);
        EXPECT_VEC_SOFT_EQ(expected.results[i].err_state.mom,
                           actual.results[i].err_state.mom);
    }
}

TEST_F(DormandPrinceStepperTest, time_shared_memory)
{
    auto old = simulate_multi_next_chord(one_thread, number_states_sample);
    auto multi_threaded
        = simulate_multi_next_chord(multi_thread, number_states_sample, true);

    CELER_VALIDATE(multi_threaded.milliseconds <= old.milliseconds,
                   << "Multi-threaded (" << multi_threaded.milliseconds
                   << " ms) is slower than old implementation ("
                   << old.milliseconds << " ms)");
}

TEST_F(DormandPrinceStepperTest, DISABLED_compare_time_one_block)
{
    constexpr int max_states_old = 768;
    constexpr int max_states_global = 192;
    constexpr int max_states_shared = 88;

    for (int i = 1; i < max_states_old; i++)
    {
        auto old = simulate_multi_next_chord(one_thread, i);
        CELER_LOG(info) << "With states=" << i
                        << " (time in ms):\tOne_thread=" << old.milliseconds;
        if (old.milliseconds == -1)
        {
            break;
        }
    }

    for (int i = 1; i < max_states_global; i++)
    {
        auto global_multi_threaded = simulate_multi_next_chord(multi_thread, i);
        CELER_LOG(info) << "With states=" << i
                        << " (time in ms):\tGlobal_memory="
                        << global_multi_threaded.milliseconds;
        if (global_multi_threaded.milliseconds == -1)
        {
            break;
        }
    }

    for (int i = 1; i < max_states_shared; i++)
    {
        auto shared_multi_threaded
            = simulate_multi_next_chord(multi_thread, i, true);
        CELER_LOG(info) << "With states=" << i
                        << " (time in ms):\tShared_memory="
                        << shared_multi_threaded.milliseconds;
        if (shared_multi_threaded.milliseconds == -1)
        {
            break;
        }
    }
}

TEST_F(DormandPrinceStepperTest, DISABLED_compare_time_multiblock)
{
    constexpr int number_max_states = 276900;
    constexpr int step_size = 100;

    for (int i = step_size; i <= number_max_states; i += step_size)
    {
        auto old = simulate_multi_next_chord(one_thread, i);
        auto global_multi_threaded = simulate_multi_next_chord(multi_thread, i);
        auto shared_multi_threaded
            = simulate_multi_next_chord(multi_thread, i, true);
        CELER_LOG(info)
            << "With " << i
            << " states (time in ms):\tOne thread=" << old.milliseconds
            << "\tGlobal memory=" << global_multi_threaded.milliseconds
            << "\tShared memory=" << shared_multi_threaded.milliseconds;
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
