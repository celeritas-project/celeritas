//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file IsotropicDistribution.test.cc
//---------------------------------------------------------------------------//
#include "random/distributions/IsotropicDistribution.hh"

#include <random>
#include "base/Range.hh"
#include "gtest/Main.hh"
#include "gtest/Test.hh"

using celeritas::IsotropicDistribution;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class IsotropicDistributionTest : public celeritas::Test
{
  protected:
    void SetUp() override {}

    std::mt19937 rng;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(IsotropicDistributionTest, bin)
{
    int num_samples = 10000;

    IsotropicDistribution<> sample_isotropic;

    std::vector<int> octant_tally(8, 0);
    for (CELER_MAYBE_UNUSED int i : celeritas::range(num_samples))
    {
        auto u = sample_isotropic(rng);

        // Make sure sampled point is on the surface of the unit sphere
        double r = std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
        ASSERT_DOUBLE_EQ(r, 1.0);

        // Tally octant
        int tally_bin = 1 * (u[0] >= 0) + 2 * (u[1] >= 0) + 4 * (u[2] >= 0);
        ASSERT_GE(tally_bin, 0);
        ASSERT_LE(tally_bin, octant_tally.size() - 1);
        ++octant_tally[tally_bin];
    }

    for (int count : octant_tally)
    {
        double octant = static_cast<double>(count) / num_samples;
        EXPECT_SOFT_NEAR(octant, 1. / 8, 0.1);
    }
}
