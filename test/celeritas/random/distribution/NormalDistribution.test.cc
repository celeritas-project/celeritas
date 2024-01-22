//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/random/distribution/NormalDistribution.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/random/distribution/NormalDistribution.hh"

#include "corecel/cont/Range.hh"

#include "DiagnosticRngEngine.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

TEST(NormalDistributionTest, bin)
{
    DiagnosticRngEngine<std::mt19937> rng;
    int num_samples = 10000;

    double mean = 0.0;
    double stddev = 1.0;

    NormalDistribution<double> sample_normal{mean, stddev};

    std::vector<int> counters(6);
    for ([[maybe_unused]] int i : range(num_samples))
    {
        double x = sample_normal(rng);
        if (x < -2.0)
            ++counters[0];
        else if (x < -1.0)
            ++counters[1];
        else if (x < 0.0)
            ++counters[2];
        else if (x < 1.0)
            ++counters[3];
        else if (x < 2.0)
            ++counters[4];
        else
            ++counters[5];
    }

    int const expected_counters[] = {235, 1379, 3397, 3411, 1352, 226};
    EXPECT_VEC_EQ(expected_counters, counters);
    EXPECT_EQ(2 * num_samples, rng.count());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
