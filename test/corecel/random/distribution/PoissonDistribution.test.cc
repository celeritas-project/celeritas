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
#include "corecel/random/HistogramSampler.hh"

#include "TestMacros.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// SampleHistogram bins doubles, but Knuth poisson returns integers
CELER_FORCEINLINE double static_cast_double(size_type v)
{
    return static_cast<double>(v);
}

//---------------------------------------------------------------------------//

TEST(PoissonDistributionKnuthTest, zero)
{
    HistogramSampler calc_histogram(8, {0, 8}, 100000);

    EXPECT_REF_EQ((SampledHistogram{{1, 0, 0, 0, 0, 0, 0, 0}, 2}),
                  calc_histogram(static_cast_double,
                                 PoissonDistributionKnuth<double>{0}));
}

TEST(PoissonDistributionKnuthTest, small)
{
    HistogramSampler calc_histogram(8, {0, 8}, 100000);
    std::vector<SampledHistogram> actual;

    for (double lambda : {0.05, 0.1, 0.2, 0.5})
    {
        PoissonDistributionKnuth<double> sample_poisson{lambda};
        actual.push_back(calc_histogram(static_cast_double, sample_poisson));
    }

    static SampledHistogram const expected[] = {
        {{0.95199, 0.04675, 0.00124, 2e-05, 0, 0, 0, 0}, 2.09858},
        {{0.90498, 0.09051, 0.00437, 0.00014, 0, 0, 0, 0}, 2.19934},
        {{0.81752, 0.1651, 0.01628, 0.00104, 6e-05, 0, 0, 0}, 2.40204},
        {{0.60662, 0.30242, 0.07653, 0.01286, 0.0014, 0.00016, 1e-05, 0},
         3.00104},
    };
    EXPECT_REF_EQ(expected, actual);
}

TEST(PoissonDistributionKnuthTest, large)
{
    HistogramSampler calc_histogram(8, {0, 50}, 1000);
    std::vector<SampledHistogram> actual;
    // Test default lambda=1
    actual.push_back(calc_histogram(static_cast_double,
                                    PoissonDistributionKnuth<double>{}));

    for (auto i : range(1, 18))
    {
        PoissonDistributionKnuth<double> sample_poisson{static_cast<double>(i)};
        actual.push_back(calc_histogram(static_cast_double, sample_poisson));
    }
    static SampledHistogram const expected[] = {
        {{0.15984, 0.00016, 0, 0, 0, 0, 0, 0}, 4.048},
        {{0.16, 0, 0, 0, 0, 0, 0, 0}, 4.036},
        {{0.15984, 0.00016, 0, 0, 0, 0, 0, 0}, 5.972},
        {{0.156, 0.004, 0, 0, 0, 0, 0, 0}, 7.74},
        {{0.14496, 0.01504, 0, 0, 0, 0, 0, 0}, 9.814},
        {{0.12064, 0.0392, 0.00016, 0, 0, 0, 0, 0}, 12.202},
        {{0.09984, 0.05856, 0.0016, 0, 0, 0, 0, 0}, 13.876},
        {{0.07392, 0.08288, 0.0032, 0, 0, 0, 0, 0}, 15.78},
        {{0.04768, 0.10112, 0.01088, 0.00032, 0, 0, 0, 0}, 18.258},
        {{0.03344, 0.10608, 0.02032, 0.00016, 0, 0, 0, 0}, 20.07},
        {{0.02144, 0.10528, 0.03248, 0.0008, 0, 0, 0, 0}, 21.976},
        {{0.01264, 0.1024, 0.04272, 0.00208, 0.00016, 0, 0, 0}, 23.454},
        {{0.00944, 0.08576, 0.05984, 0.00496, 0, 0, 0, 0}, 25.636},
        {{0.00336, 0.0696, 0.0752, 0.0112, 0.00064, 0, 0, 0}, 28.374},
        {{0.00208, 0.05632, 0.08352, 0.01744, 0.00064, 0, 0, 0}, 29.918},
        {{0.00128, 0.04288, 0.0824, 0.03104, 0.00224, 0.00016, 0, 0}, 32.384},
        {{0.0008, 0.0288, 0.08672, 0.03984, 0.00384, 0, 0, 0}, 34.208},
        {{0.00032, 0.02192, 0.08336, 0.04736, 0.00672, 0.00032, 0, 0}, 36.12},
    };
    EXPECT_REF_EQ(expected, actual) << repr(actual);
}

TEST(PoissonDistributionTest, bin_zero)
{
    int num_samples = 100;

    // Small lambda will use the direct method, which requires on average
    // lambda + 1 RNG samples
    PoissonDistribution<double> sample_poisson{0};
    DiagnosticRngEngine<std::mt19937> rng;

    Histogram histogram(4, {0, 1e-3});
    for ([[maybe_unused]] int i : range(num_samples))
    {
        histogram(static_cast_double(sample_poisson(rng)));
    }
    static unsigned int const expected_counts[] = {100, 0, 0, 0};
    EXPECT_VEC_EQ(expected_counts, histogram.counts());
    EXPECT_EQ(0, rng.count());
}

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
        histogram(static_cast_double(sample_poisson(rng)));
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
        histogram(static_cast_double(sample_poisson(rng)));
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
