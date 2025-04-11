//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/distribution/PowerDistribution.test.cc
//---------------------------------------------------------------------------//
#include "corecel/random/distribution/PowerDistribution.hh"

#include "corecel/random/Histogram.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

TEST(PowerDistributionTest, squared)
{
    constexpr int num_samples = 10000;

    // Sample x^2 from 0 to 1
    PowerDistribution<double> sample{2};
    std::mt19937 rng;

    Histogram histogram(10, {0, 1});
    for (int i = 0; i < num_samples; ++i)
    {
        double r = sample(rng);
        histogram(r);
    }

    static size_type const expected_counts[] = {
        8ul, 72ul, 195ul, 364ul, 658ul, 950ul, 1219ul, 1641ul, 2253ul, 2640ul};
    EXPECT_VEC_EQ(expected_counts, histogram.counts());
    EXPECT_EQ(num_samples, histogram.calc_valid_samples());
}

//---------------------------------------------------------------------------//

TEST(PowerDistributionTest, positive)
{
    constexpr int num_samples = 10000;

    PowerDistribution<double> sample{2.25, 1.5, 3.5};
    std::mt19937 rng;

    Histogram histogram(10, {1.5, 3.5});
    for (int i = 0; i < num_samples; ++i)
    {
        double r = sample(rng);
        histogram(r);
    }

    static size_type const expected_counts[] = {
        344ul, 459ul, 623ul, 698ul, 806ul, 1004ul, 1166ul, 1459ul, 1604ul, 1837ul};
    EXPECT_VEC_EQ(expected_counts, histogram.counts());
    EXPECT_EQ(num_samples, histogram.calc_valid_samples());
}

//---------------------------------------------------------------------------//

TEST(PowerDistributionTest, negative)
{
    constexpr int num_samples = 10000;

    PowerDistribution<double> sample{-2.5, 0.1, 10.1};
    std::mt19937 rng;

    Histogram histogram(10, {0.1, 10.1});
    for (int i = 0; i < num_samples; ++i)
    {
        double r = sample(rng);
        histogram(r);
    }

    static unsigned int const expected_counts[]
        = {9729ul, 171ul, 60ul, 17ul, 6ul, 8ul, 3ul, 3ul, 1ul, 2ul};
    EXPECT_VEC_EQ(expected_counts, histogram.counts());
    EXPECT_EQ(num_samples, histogram.calc_valid_samples());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
