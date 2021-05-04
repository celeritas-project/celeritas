//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ReciprocalDistribution.test.cc
//---------------------------------------------------------------------------//
#include "random/distributions/ReciprocalDistribution.hh"

#include "base/Range.hh"
#include "celeritas_test.hh"

using celeritas::ReciprocalDistribution;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class ReciprocalDistributionTest : public celeritas::Test
{
  protected:
    void SetUp() override {}

    std::mt19937 rng;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(ReciprocalDistributionTest, bin)
{
    int num_samples = 10000;

    double min = 0.1;
    double max = 0.9;

    ReciprocalDistribution<double> sample_recip{min, max};

    std::vector<int> counters(10);
    for (CELER_MAYBE_UNUSED int i : celeritas::range(num_samples))
    {
        double r = sample_recip(rng);
        ASSERT_GE(r, min);
        ASSERT_LE(r, max);
        int bin = int(1.0 / r);
        CELER_ASSERT(bin >= 0 && bin < static_cast<int>(counters.size()));
        counters[bin] += 1;
    }

    const int expected_counters[]
        = {0, 2601, 1905, 1324, 974, 771, 747, 630, 582, 466};
    EXPECT_VEC_EQ(expected_counters, counters);
}
