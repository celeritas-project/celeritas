//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/distribution/GammaDistribution.test.cc
//---------------------------------------------------------------------------//
#include "corecel/random/distribution/GammaDistribution.hh"

#include "corecel/cont/Range.hh"
#include "corecel/random/DiagnosticRngEngine.hh"
#include "corecel/random/Histogram.hh"

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

    Histogram histogram(8, {0, 8});
    for ([[maybe_unused]] int i : range(num_samples))
    {
        histogram(sample_gamma(rng));
    }
    static unsigned int const expected_counts[]
        = {2, 209, 1387, 2529, 2548, 1784, 916, 413};
    EXPECT_VEC_EQ(expected_counts, histogram.counts());
    EXPECT_EQ(40118, rng.count());
}

TEST_F(GammaDistributionTest, bin_small_alpha)
{
    int num_samples = 10000;

    double alpha = 0.5;
    double beta = 1.0;
    GammaDistribution<double> sample_gamma{alpha, beta};

    Histogram histogram(8, {0, 8});
    for ([[maybe_unused]] int i : range(num_samples))
    {
        histogram(sample_gamma(rng));
    }
    static unsigned int const expected_counts[]
        = {8486, 1081, 310, 79, 28, 11, 1, 1};
    EXPECT_VEC_EQ(expected_counts, histogram.counts());
    EXPECT_EQ(61136, rng.count());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
