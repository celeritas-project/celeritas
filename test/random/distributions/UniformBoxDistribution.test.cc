//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file UniformBoxDistribution.test.cc
//---------------------------------------------------------------------------//
#include "random/distributions/UniformBoxDistribution.hh"

#include <random>

#include "base/Range.hh"

#include "../DiagnosticRngEngine.hh"
#include "celeritas_test.hh"

using celeritas::UniformBoxDistribution;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class UniformBoxDistributionTest : public celeritas::Test
{
  protected:
    void SetUp() override {}

    celeritas_test::DiagnosticRngEngine<std::mt19937> rng;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(UniformBoxDistributionTest, all)
{
    int num_samples = 10000;

    UniformBoxDistribution<> sample_point({-1, -2, -3}, {3, 6, 9});

    std::vector<int> octant_tally(8, 0);
    for (CELER_MAYBE_UNUSED int i : celeritas::range(num_samples))
    {
        auto r = sample_point(rng);

        // Make sure sampled point is in bounds
        ASSERT_TRUE(r[0] >= -1 && r[0] <= 3 && r[1] >= -2 && r[1] <= 6
                    && r[2] >= -3 && r[2] <= 9);

        // Tally octant
        int tally_bin = 1 * (r[0] >= 1) + 2 * (r[1] >= 2) + 4 * (r[2] >= 3);
        ASSERT_TRUE(tally_bin >= 0 && tally_bin < int(octant_tally.size()));
        ++octant_tally[tally_bin];
    }

    for (int count : octant_tally)
    {
        double octant = static_cast<double>(count) / num_samples;
        EXPECT_SOFT_NEAR(octant, 1. / 8, 0.1);
    }
    // 2 32-bit samples per double, 3 doubles per sample
    EXPECT_EQ(num_samples * 6, rng.count());
}
