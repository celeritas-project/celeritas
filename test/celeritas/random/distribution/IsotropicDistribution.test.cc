//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/random/distribution/IsotropicDistribution.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/random/distribution/IsotropicDistribution.hh"

#include <random>

#include "corecel/cont/Range.hh"
#include "corecel/math/ArrayUtils.hh"

#include "DiagnosticRngEngine.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
TEST(IsotropicDistributionTest, bin)
{
    int num_samples = 10000;

    IsotropicDistribution<double> sample_isotropic;
    test::DiagnosticRngEngine<std::mt19937> rng;

    std::vector<int> octant_tally(8, 0);
    for ([[maybe_unused]] int i : range(num_samples))
    {
        auto u = sample_isotropic(rng);

        // Make sure sampled point is on the surface of the unit sphere
        ASSERT_TRUE(is_soft_unit_vector(u));

        // Tally octant
        int tally_bin = 1 * (u[0] >= 0) + 2 * (u[1] >= 0) + 4 * (u[2] >= 0);
        ASSERT_GE(tally_bin, 0);
        ASSERT_LT(tally_bin, octant_tally.size());
        ++octant_tally[tally_bin];
    }

    for (int count : octant_tally)
    {
        double octant = static_cast<double>(count) / num_samples;
        EXPECT_SOFT_NEAR(octant, 1. / 8, 0.1);
    }
    // 2 32-bit samples per double, 2 doubles per sample
    EXPECT_EQ(num_samples * 4, rng.count());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
