//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/random/distribution/GammaDistribution.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/random/distribution/GammaDistribution.hh"

#include "corecel/cont/Range.hh"

#include "DiagnosticRngEngine.hh"
#include "celeritas_test.hh"

using celeritas::GammaDistribution;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class GammaDistributionTest : public celeritas_test::Test
{
  protected:
    void SetUp() override {}

    celeritas_test::DiagnosticRngEngine<std::mt19937> rng;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(GammaDistributionTest, bin_large_alpha)
{
    int num_samples = 10000;

    double alpha = 9.0;
    double beta  = 0.5;

    GammaDistribution<double> sample_gamma{alpha, beta};

    std::vector<int> counters(7);
    for (CELER_MAYBE_UNUSED int i : celeritas::range(num_samples))
    {
        double x = sample_gamma(rng);
        if (x < 2.0)
            ++counters[0];
        else if (x < 3.0)
            ++counters[1];
        else if (x < 4.0)
            ++counters[2];
        else if (x < 5.0)
            ++counters[3];
        else if (x < 6.0)
            ++counters[4];
        else if (x < 7.0)
            ++counters[5];
        else
            ++counters[6];
    }

    const int expected_counters[] = {211, 1387, 2529, 2548, 1784, 916, 625};
    EXPECT_VEC_EQ(expected_counters, counters);
    EXPECT_EQ(40118, rng.count());
}

TEST_F(GammaDistributionTest, bin_small_alpha)
{
    int num_samples = 10000;

    double alpha = 0.5;
    double beta  = 1.0;

    GammaDistribution<double> sample_gamma{alpha, beta};

    std::vector<int> counters(5);
    for (CELER_MAYBE_UNUSED int i : celeritas::range(num_samples))
    {
        double x = sample_gamma(rng);
        if (x < 1.0)
            ++counters[0];
        else if (x < 2.0)
            ++counters[1];
        else if (x < 3.0)
            ++counters[2];
        else if (x < 4.0)
            ++counters[3];
        else
            ++counters[4];
    }

    const int expected_counters[] = {8486, 1081, 310, 79, 44};
    EXPECT_VEC_EQ(expected_counters, counters);
    EXPECT_EQ(61136, rng.count());
}
