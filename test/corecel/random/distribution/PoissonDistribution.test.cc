//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/distribution/PoissonDistribution.test.cc
//---------------------------------------------------------------------------//
#include "corecel/random/distribution/PoissonDistribution.hh"

#include <map>

#include "corecel/cont/Range.hh"
#include "corecel/random/DiagnosticRngEngine.hh"

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

    std::map<int, int> sample_to_count;
    for ([[maybe_unused]] int i : range(num_samples))
    {
        auto k = sample_poisson(rng);
        ++sample_to_count[k];
    }

    std::vector<int> samples;
    std::vector<int> counts;
    for (auto const& it : sample_to_count)
    {
        samples.push_back(it.first);
        counts.push_back(it.second);
    }

    int const expected_samples[]
        = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    int const expected_counts[] = {
        177, 762, 1444, 1971, 1950, 1586, 1054, 562, 286, 125, 55, 18, 5, 1, 3, 1};
    EXPECT_VEC_EQ(expected_samples, samples);
    EXPECT_VEC_EQ(expected_counts, counts);
    EXPECT_EQ(99684, rng.count());
}

TEST(PoissonDistributionTest, bin_large)
{
    int num_samples = 10000;

    // Large lambda will use Gaussian approximation
    double lambda = 64.0;
    PoissonDistribution<double> sample_poisson{lambda};
    DiagnosticRngEngine<std::mt19937> rng;

    std::map<int, int> sample_to_count;
    for ([[maybe_unused]] int i : range(num_samples))
    {
        auto k = sample_poisson(rng);
        ++sample_to_count[k];
    }

    std::vector<int> samples;
    std::vector<int> counts;
    for (auto const& it : sample_to_count)
    {
        samples.push_back(it.first);
        counts.push_back(it.second);
    }

    int const expected_samples[]
        = {35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
           50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
           65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
           80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 94};
    int const expected_counts[]
        = {1,   1,   5,   2,   5,   6,   6,   11,  11,  11,  28,  45,
           58,  80,  72,  123, 135, 157, 203, 218, 272, 315, 352, 382,
           389, 442, 454, 470, 508, 502, 490, 504, 438, 456, 410, 363,
           337, 301, 239, 220, 187, 160, 161, 114, 95,  65,  57,  38,
           34,  22,  11,  8,   8,   6,   6,   1,   2,   2,   1};
    EXPECT_VEC_EQ(expected_samples, samples);
    EXPECT_VEC_EQ(expected_counts, counts);
    EXPECT_EQ(2 * num_samples, rng.count());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
