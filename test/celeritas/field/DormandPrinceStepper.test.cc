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
TEST_F(DormandPrinceStepperTest, global_memory_result)
{
    auto expected = simulate_multi_next_chord(one_thread, number_states_sample);
    auto actual = simulate_multi_next_chord(multi_thread, number_states_sample);

    for (int i = 0; i < number_states_sample; i++)
    {
        // CELER_LOG(info) << "Result for state " << std::to_string(i) << ":\n"
        //                 << print_results(expected.results[i],
        //                                  actual.results[i]);

        CELER_VALIDATE(
            compare_results(expected.results[i], actual.results[i]),
            << "Error at state " << std::to_string(i) << "\n"
            << print_results(expected.results[i], actual.results[i]));
    }
}

TEST_F(DormandPrinceStepperTest, global_memory_time)
{
    auto old = simulate_multi_next_chord(one_thread, number_states_sample);
    auto multi_threaded
        = simulate_multi_next_chord(multi_thread, number_states_sample);

    CELER_VALIDATE(multi_threaded.milliseconds <= old.milliseconds,
                   << "Multi-threaded (" << multi_threaded.milliseconds
                   << " ms) is not faster than old implementation ("
                   << old.milliseconds << " ms)");
}

TEST_F(DormandPrinceStepperTest, shared_memory_result)
{
    auto expected = simulate_multi_next_chord(one_thread, number_states_sample);
    auto actual
        = simulate_multi_next_chord(multi_thread, number_states_sample, true);

    for (int i = 0; i < number_states_sample; i++)
    {
        CELER_VALIDATE(
            compare_results(expected.results[i], actual.results[i]),
            << "Error at state " << std::to_string(i) << "\n"
            << print_results(expected.results[i], actual.results[i]));
    }
}

TEST_F(DormandPrinceStepperTest, shared_memory_time)
{
    auto old = simulate_multi_next_chord(one_thread, number_states_sample);
    auto multi_threaded
        = simulate_multi_next_chord(multi_thread, number_states_sample, true);

    CELER_VALIDATE(multi_threaded.milliseconds <= old.milliseconds,
                   << "Multi-threaded (" << multi_threaded.milliseconds
                   << " ms) is not faster than old implementation ("
                   << old.milliseconds << " ms)");
}

TEST_F(DormandPrinceStepperTest, compare_time)
{
    constexpr int number_max_states = 427;
#define debug
#ifdef debug
    for (int i = 1; i < number_max_states; i++)
    {
        auto old = simulate_multi_next_chord(one_thread, i);
        CELER_LOG(info) << "With states=" << i
                        << " (time in ms):\tOne_thread=" << old.milliseconds;
        if (old.milliseconds == -1)
        {
            break;
        }
    }
    for (int i = 1; i < number_max_states; i++)
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
    for (int i = 1; i < number_max_states; i++)
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

#else

    for (int i = 1; i < number_max_states; i++)
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
#endif
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
