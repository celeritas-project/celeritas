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

    std::vector<int> counters(5);
    for ([[maybe_unused]] int i : range(num_samples))
    {
        double x = sample(rng);
        ASSERT_GE(x, 0.0);
        if (x < 1.0)
            ++counters[0];
        else if (x < 2.0)
            ++counters[1];
        else if (x < 4.0)
            ++counters[2];
        else if (x < 8.0)
            ++counters[3];
        else
            ++counters[4];
    }

    // PRINT_EXPECTED(counters);
    int const expected_counters[] = {2180, 1717, 2411, 2265, 1427};
    EXPECT_VEC_EQ(expected_counters, counters);
    EXPECT_EQ(2 * num_samples, rng.count());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
