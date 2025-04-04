//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/distribution/ReciprocalDistribution.test.cc
//---------------------------------------------------------------------------//
#include "corecel/random/distribution/ReciprocalDistribution.hh"

#include "corecel/cont/Range.hh"
#include "corecel/random/Histogram.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
TEST(ReciprocalDistributionTest, bin)
{
    int num_samples = 10000;

    double min = 0.1;
    double max = 0.9;

    ReciprocalDistribution<double> sample_recip{min, max};
    std::mt19937 rng;

    Histogram histogram(10, {0, 10});
    for ([[maybe_unused]] int i : range(num_samples))
    {
        double r = sample_recip(rng);
        ASSERT_GE(r, min);
        ASSERT_LE(r, max);
        histogram(1.0 / r);
    }

    static unsigned int const expected_counts[]
        = {0, 2601, 1905, 1324, 974, 771, 747, 630, 582, 466};
    EXPECT_VEC_EQ(expected_counts, histogram.counts());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
