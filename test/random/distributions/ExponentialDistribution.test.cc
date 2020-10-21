//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ExponentialDistribution.test.cc
//---------------------------------------------------------------------------//
#include "random/distributions/ExponentialDistribution.hh"

#include <random>
#include "base/Range.hh"
#include "gtest/Main.hh"
#include "gtest/Test.hh"

using celeritas::ExponentialDistribution;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class ExponentialDistributionTest : public celeritas::Test
{
  protected:
    void SetUp() override {}

    std::mt19937 rng;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(ExponentialDistributionTest, all)
{
    int                       num_samples = 10000;
    double                    lambda      = 0.25;
    ExponentialDistribution<> sample(lambda);

    std::vector<int> counters(5);
    for (CELER_MAYBE_UNUSED int i : celeritas::range(num_samples))
    {
        double x = sample(rng);
        ASSERT_GE(x, 0.0);
        if (x < 1.0)
            ++counters[0];
        else if (x < 2.0)
            ++counters[1];
        else if (x < 4.0)
            ++counters[2];
        else if (x < 8.0)
            ++counters[3];
        else
            ++counters[4];
    }

    // PRINT_EXPECTED(counters);
    const int expected_counters[] = {2180, 1717, 2411, 2265, 1427};
    EXPECT_VEC_EQ(expected_counters, counters);
}
