//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/distribution/RadialDistribution.test.cc
//---------------------------------------------------------------------------//
#include "corecel/random/distribution/RadialDistribution.hh"

#include <random>

#include "corecel/cont/Range.hh"
#include "corecel/random/DiagnosticRngEngine.hh"
#include "corecel/random/Histogram.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

TEST(RadialDistributionTest, bin)
{
    int num_samples = 10000;

    double radius = 5.0;
    RadialDistribution<> sample_radial(radius);
    DiagnosticRngEngine<std::mt19937> rng;

    Histogram histogram(5, {0, 5});
    for ([[maybe_unused]] int i : range(num_samples))
    {
        double r = sample_radial(rng);
        ASSERT_GE(r, 0.0);
        ASSERT_LE(r, radius);
        histogram(r);
    }

    if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        static unsigned int const expected_counts[]
            = {80, 559, 1608, 2860, 4893};
        EXPECT_VEC_EQ(expected_counts, histogram.counts());
    }
    EXPECT_EQ(num_samples * (sizeof(real_type) / 4), rng.count());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
