//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/distribution/PoissonDistribution.test.cc
//---------------------------------------------------------------------------//
#include "corecel/random/distribution/PoissonDistribution.hh"

#include "corecel/cont/Range.hh"
#include "corecel/random/DiagnosticRngEngine.hh"
#include "corecel/random/Histogram.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

TEST(PoissonDistributionTest, bin_small)
{
    int num_samples = 10000;

    // Small lambda will use the direct method, which requires on average
    // lambda + 1 RNG samples
    double lambda = 4.0;
    PoissonDistribution<double> sample_poisson{lambda};
    DiagnosticRngEngine<std::mt19937> rng;

    Histogram histogram(16, {0, 16});
    for ([[maybe_unused]] int i : range(num_samples))
    {
        histogram(sample_poisson(rng));
    }
    static unsigned int const expected_counts[] = {
        177, 762, 1444, 1971, 1950, 1586, 1054, 562, 286, 125, 55, 18, 5, 1, 3, 1};
    EXPECT_VEC_EQ(expected_counts, histogram.counts());
    EXPECT_EQ(99684, rng.count());
}

TEST(PoissonDistributionTest, bin_large)
{
    int num_samples = 10000;

    // Large lambda will use Gaussian approximation
    double lambda = 64.0;
    PoissonDistribution<double> sample_poisson{lambda};
    DiagnosticRngEngine<std::mt19937> rng;

    // Since the result are integers, bin centers should be integer
    Histogram histogram(60, {34.5, 94.5});
    for ([[maybe_unused]] int i : range(num_samples))
    {
        histogram(static_cast<double>(sample_poisson(rng)));
    }
    static unsigned int const expected_counts[]
        = {1,   1,   5,   2,   5,   6,   6,   11,  11,  11,  28,  45,
           58,  80,  72,  123, 135, 157, 203, 218, 272, 315, 352, 382,
           389, 442, 454, 470, 508, 502, 490, 504, 438, 456, 410, 363,
           337, 301, 239, 220, 187, 160, 161, 114, 95,  65,  57,  38,
           34,  22,  11,  8,   8,   6,   6,   1,   2,   2,   0,   1};
    EXPECT_VEC_EQ(expected_counts, histogram.counts());
    EXPECT_EQ(2 * num_samples, rng.count());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
