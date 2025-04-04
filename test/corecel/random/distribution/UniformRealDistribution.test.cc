//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/distribution/UniformRealDistribution.test.cc
//---------------------------------------------------------------------------//
#include "corecel/random/distribution/UniformRealDistribution.hh"

#include <random>

#include "corecel/cont/Range.hh"
#include "corecel/random/Histogram.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class UniformRealDistributionTest : public Test
{
  protected:
    std::mt19937 rng;
};

TEST_F(UniformRealDistributionTest, constructors)
{
    {
        UniformRealDistribution<> sample_uniform{};
        EXPECT_SOFT_EQ(0.0, sample_uniform.a());
        EXPECT_SOFT_EQ(1.0, sample_uniform.b());
    }
    {
        UniformRealDistribution<> sample_uniform{1, 2};
        EXPECT_SOFT_EQ(1.0, sample_uniform.a());
        EXPECT_SOFT_EQ(2.0, sample_uniform.b());
    }
    if (CELERITAS_DEBUG)
    {
        // b < a is not allowed
        EXPECT_THROW(UniformRealDistribution<>(3, 2), DebugError);
    }
}

TEST_F(UniformRealDistributionTest, bin)
{
    int num_samples = 10000;

    double min = 0.0;
    double max = 5.0;
    UniformRealDistribution<double> sample_uniform{min, max};

    Histogram histogram(5, {0, 5});
    for ([[maybe_unused]] int i : range(num_samples))
    {
        double r = sample_uniform(rng);
        ASSERT_GE(r, min);
        ASSERT_LE(r, max);
        histogram(r);
    }

    static unsigned int const expected_counts[]
        = {2071, 1955, 1991, 2013, 1970};
    EXPECT_VEC_EQ(expected_counts, histogram.counts());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
