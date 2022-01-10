//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RadialDistribution.test.cc
//---------------------------------------------------------------------------//
#include "random/distributions/RadialDistribution.hh"

#include <random>
#include "base/Range.hh"
#include "celeritas_test.hh"

using celeritas::RadialDistribution;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class RadialDistributionTest : public celeritas::Test
{
  protected:
    void SetUp() override {}

    std::mt19937 rng;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(RadialDistributionTest, bin)
{
    int num_samples = 10000;

    double               radius = 5.0;
    RadialDistribution<> sample_radial(radius);

    std::vector<int> counters(5);
    for (CELER_MAYBE_UNUSED int i : celeritas::range(num_samples))
    {
        double r = sample_radial(rng);
        ASSERT_GE(r, 0.0);
        ASSERT_LE(r, radius);
        counters[int(r)] += 1;
    }

    // PRINT_EXPECTED(counters);
    const int expected_counters[] = {80, 559, 1608, 2860, 4893};
    EXPECT_VEC_EQ(expected_counters, counters);
}
