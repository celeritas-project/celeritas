//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file UniformRealDistribution.test.cc
//---------------------------------------------------------------------------//
#include "random/UniformRealDistribution.hh"

#include <random>
#include "base/Range.hh"
#include "gtest/Main.hh"
#include "gtest/Test.hh"

using celeritas::UniformRealDistribution;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class UniformRealDistributionTest : public celeritas::Test
{
  protected:
    void SetUp() override {}

    std::mt19937 rng;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(UniformRealDistributionTest, bin)
{
    int num_samples = 10000;

    double min = 0.0;
    double max = 5.0;
    UniformRealDistribution<> sample_uniform{min, max};

    std::vector<int> counters(5);
    for (int i : celeritas::range(num_samples))
    {
        double r = sample_uniform(rng);
        ASSERT_GE(r, min);
        ASSERT_LE(r, max);
        counters[int(r)] += 1;
    }

    for (int count : counters)
    {
        cout << count << ' ';
    }
    cout << endl;
}
