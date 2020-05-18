//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RadialDistribution.test.cc
//---------------------------------------------------------------------------//
#include "random/RadialDistribution.hh"

#include <random>
#include "base/Range.hh"
#include "gtest/Main.hh"
#include "gtest/Test.hh"

using celeritas::RadialDistribution;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class RadialDistributionTest : public celeritas::Test
{
  protected:
    void SetUp() override {}

    std::mt19937 rng;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(RadialDistributionTest, bin)
{
    RadialDistribution<> sample_radius(5.0);

    std::vector<int> counters(5);
    for (int i : celeritas::range(10000))
    {
        double r = sample_radius(rng);
        ASSERT_GE(r, 0);
        ASSERT_LE(r, 5.0);
        counters[int(r)] += 1;
    }

    for (int count : counters)
    {
        cout << count << ' ';
    }
    cout << endl;
}
