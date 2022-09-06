//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/random/distribution/RadialDistribution.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/random/distribution/RadialDistribution.hh"

#include <random>

#include "corecel/cont/Range.hh"

#include "DiagnosticRngEngine.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

TEST(RadialDistributionTest, bin)
{
    int num_samples = 10000;

    double                            radius = 5.0;
    RadialDistribution<>              sample_radial(radius);
    DiagnosticRngEngine<std::mt19937> rng;

    std::vector<int> counters(5);
    for (CELER_MAYBE_UNUSED int i : range(num_samples))
    {
        double r = sample_radial(rng);
        ASSERT_GE(r, 0.0);
        ASSERT_LE(r, radius);
        counters[int(r)] += 1;
    }

    const int expected_counters[] = {80, 559, 1608, 2860, 4893};
    EXPECT_VEC_EQ(expected_counters, counters);
    EXPECT_EQ(2 * num_samples, rng.count());
}

//---------------------------------------------------------------------------//
} // namespace test
} // namespace celeritas
