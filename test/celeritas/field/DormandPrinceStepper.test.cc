//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/DormandPrinceStepper.cc
//---------------------------------------------------------------------------//
#include "DormandPrinceStepper.test.hh"

#include "celeritas_test.hh"   // for CELER_EXPECT and Test

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
// TEST_F(DormandPrinceStepperTest, test)
// {
//     test();
//     CELER_EXPECT(true);
// }

TEST_F(DormandPrinceStepperTest, gpu_result)
{

    auto expected = simulate_multi_next_chord(one_thread);
    auto actual = simulate_multi_next_chord(multi_thread);

    for (int i = 0; i < number_of_states; i++)
    {
        // CELER_LOG(info) << "Result for state " << std::to_string(i) << ":\n"
        //                 << print_results(expected.results[i],
        //                                  actual.results[i]);

        CELER_VALIDATE(compare_results(expected.results[i], actual.results[i]),
                    << "Error at state " << std::to_string(i) << "\n" 
                    << print_results(expected.results[i], actual.results[i]));
    }

}

TEST_F(DormandPrinceStepperTest, gpu_time)
{
    auto old = simulate_multi_next_chord(one_thread);
    CELER_LOG(info) << "Old time: " << old.milliseconds << " ms";

    auto multi_threaded = simulate_multi_next_chord(multi_thread);
    CELER_LOG(info) << "Multi-threaded time: " << multi_threaded.milliseconds
                    << " ms";

    CELER_VALIDATE(multi_threaded.milliseconds <= old.milliseconds,
                   << "Multi-threaded is not faster than old implementation");
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
