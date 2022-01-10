//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file NormalDistribution.test.cc
//---------------------------------------------------------------------------//
#include "random/distributions/NormalDistribution.hh"

#include "base/Range.hh"
#include "celeritas_test.hh"
#include "../DiagnosticRngEngine.hh"

using celeritas::NormalDistribution;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class NormalDistributionTest : public celeritas::Test
{
  protected:
    void SetUp() override {}

    celeritas_test::DiagnosticRngEngine<std::mt19937> rng;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(NormalDistributionTest, bin)
{
    int num_samples = 10000;

    double mean   = 0.0;
    double stddev = 1.0;

    NormalDistribution<double> sample_normal{mean, stddev};

    std::vector<int> counters(6);
    for (CELER_MAYBE_UNUSED int i : celeritas::range(num_samples))
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

    const int expected_counters[] = {235, 1379, 3397, 3411, 1352, 226};
    EXPECT_VEC_EQ(expected_counters, counters);
    EXPECT_EQ(2 * num_samples, rng.count());
}
