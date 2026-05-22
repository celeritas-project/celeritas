//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/distribution/TruncatedDistribution.test.cc
//---------------------------------------------------------------------------//
#include "corecel/random/distribution/TruncatedDistribution.hh"

#include <random>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/random/DiagnosticRngEngine.hh"
#include "corecel/random/Histogram.hh"
#include "corecel/random/HistogramSampler.hh"
#include "corecel/random/distribution/NormalDistribution.hh"
#include "corecel/random/distribution/UniformRealDistribution.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

TEST(TruncatedDistributionTest, normal)
{
    using TruncatedNormal = TruncatedDistribution<NormalDistribution<double>>;

    DiagnosticRngEngine<std::mt19937> rng;
    int num_samples = 10000;

    double mean = 0.0;
    double stddev = 1.0;
    double lower = -2;
    double upper = 2;
    TruncatedNormal sample_normal{lower, upper, mean, stddev};

    Histogram histogram(8, {lower, upper});
    for ([[maybe_unused]] int i : range(num_samples))
    {
        histogram(sample_normal(rng));
    }
    static unsigned int const expected_counts[]
        = {449u, 1008u, 1567u, 1996u, 1992u, 1567u, 917u, 504u};
    EXPECT_VEC_EQ(expected_counts, histogram.counts());
    EXPECT_EQ(0, histogram.underflow());
    EXPECT_EQ(0, histogram.overflow());
    EXPECT_EQ(20984, rng.count());
}

TEST(TruncatedDistributionTest, uniform)
{
    using TruncatedUniformDbl
        = TruncatedDistribution<UniformRealDistribution<double>>;

    // Check that all values are within truncated range
    HistogramSampler calc_histogram(8, {0, 4}, 100);
    // Underlying distribution samples between -1, 4
    auto result = calc_histogram(TruncatedUniformDbl(0, 4, -1, 4));

    SampledHistogram ref;
    ref.distribution = {0.16, 0.26, 0.28, 0.26, 0.14, 0.34, 0.24, 0.32};
    ref.rng_count = 2.36;
    EXPECT_REF_EQ(ref, result);
}

TEST(TruncatedDistributionTest, TEST_IF_CELERITAS_DEBUG(rejection))
{
    using TruncatedUniformDbl
        = TruncatedDistribution<UniformRealDistribution<double>>;
    // Truncated bounds are outside actual distribution
    TruncatedUniformDbl sample_forever(-10, -1, 1, 5);

    DiagnosticRngEngine<std::mt19937> rng;
    EXPECT_THROW(sample_forever(rng), DebugError);
    EXPECT_EQ(64, rng.exchange_count());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
