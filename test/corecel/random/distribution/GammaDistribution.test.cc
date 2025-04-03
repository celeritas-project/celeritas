//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/distribution/GammaDistribution.test.cc
//---------------------------------------------------------------------------//
#include "corecel/random/distribution/GammaDistribution.hh"

#include "corecel/cont/Range.hh"
#include "corecel/random/DiagnosticRngEngine.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
class GammaDistributionTest : public Test
{
  protected:
    test::DiagnosticRngEngine<std::mt19937> rng;
};

TEST_F(GammaDistributionTest, bin_large_alpha)
{
    int num_samples = 10000;

    double alpha = 9.0;
    double beta = 0.5;

    GammaDistribution<double> sample_gamma{alpha, beta};

    std::vector<int> counters(7);
    for ([[maybe_unused]] int i : range(num_samples))
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

    int const expected_counters[] = {211, 1387, 2529, 2548, 1784, 916, 625};
    EXPECT_VEC_EQ(expected_counters, counters);
    EXPECT_EQ(40118, rng.count());
}

TEST_F(GammaDistributionTest, bin_small_alpha)
{
    int num_samples = 10000;

    double alpha = 0.5;
    double beta = 1.0;

    GammaDistribution<double> sample_gamma{alpha, beta};

    std::vector<int> counters(5);
    for ([[maybe_unused]] int i : range(num_samples))
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

    int const expected_counters[] = {8486, 1081, 310, 79, 44};
    EXPECT_VEC_EQ(expected_counters, counters);
    EXPECT_EQ(61136, rng.count());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
