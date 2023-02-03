//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/random/distribution/UniformRealDistribution.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/random/distribution/UniformRealDistribution.hh"

#include <random>

#include "corecel/cont/Range.hh"

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
    UniformRealDistribution<> sample_uniform{min, max};

    std::vector<int> counters(5);
    for ([[maybe_unused]] int i : range(num_samples))
    {
        double r = sample_uniform(rng);
        ASSERT_GE(r, min);
        ASSERT_LE(r, max);
        counters[int(r)] += 1;
    }

    // PRINT_EXPECTED(counters);
    int const expected_counters[] = {2071, 1955, 1991, 2013, 1970};
    EXPECT_VEC_EQ(expected_counters, counters);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
