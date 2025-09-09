//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/distribution/ExponentialDistribution.test.cc
//---------------------------------------------------------------------------//
#include "corecel/random/distribution/ExponentialDistribution.hh"

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
TEST(ExponentialDistributionTest, all)
{
    int num_samples = 10000;
    double lambda = 0.25;
    ExponentialDistribution<double> sample(lambda);
    test::DiagnosticRngEngine<std::mt19937> rng;

    Histogram histogram(8, {0, 16});
    for ([[maybe_unused]] int i : range(num_samples))
    {
        histogram(sample(rng));
    }
    static unsigned int const expected_counts[]
        = {3897u, 2411u, 1368u, 897u, 587u, 354u, 184u, 127u};
    EXPECT_VEC_EQ(expected_counts, histogram.counts());
    EXPECT_EQ(2 * num_samples, rng.count());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
