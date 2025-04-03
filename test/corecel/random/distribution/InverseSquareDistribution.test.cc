//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/distribution/InverseSquareDistribution.test.cc
//---------------------------------------------------------------------------//
#include "corecel/random/distribution/InverseSquareDistribution.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
TEST(InverseSquareDistributionTest, bin)
{
    int num_samples = 9000;

    double min = 0.1;
    double max = 0.9;

    InverseSquareDistribution<double> sample_esq{min, max};
    std::mt19937 rng;

    std::vector<int> counters(10);
    for ([[maybe_unused]] int i : range(num_samples))
    {
        double r = sample_esq(rng);
        ASSERT_GE(r, min);
        ASSERT_LE(r, max);
        int bin = int(1.0 / r);
        CELER_ASSERT(bin >= 0 && bin < static_cast<int>(counters.size()));
        ++counters[bin];
    }

    static int const expected_counters[]
        = {0, 944, 1043, 959, 972, 1027, 1045, 981, 1009, 1020};
    EXPECT_VEC_EQ(expected_counters, counters);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
