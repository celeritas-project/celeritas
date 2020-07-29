//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file BernoulliDistribution.test.cc
//---------------------------------------------------------------------------//
#include "random/distributions/BernoulliDistribution.hh"

#include <random>
#include "gtest/Main.hh"
#include "gtest/Test.hh"
#include "base/Range.hh"

using celeritas::BernoulliDistribution;

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST(BernoulliDistributionTest, single_constructor)
{
    std::mt19937          rng;
    BernoulliDistribution quarter_true(0.25);
    EXPECT_SOFT_EQ(0.25, quarter_true.p());

    int num_true = 0;
    for (CELER_MAYBE_UNUSED auto i : celeritas::range(1000))
    {
        if (quarter_true(rng))
        {
            ++num_true;
        }
    }
    EXPECT_EQ(254, num_true);
}

TEST(BernoulliDistributionTest, normalizing_constructor)
{
    BernoulliDistribution tenth_true(1, 9);
    EXPECT_SOFT_EQ(0.1, tenth_true.p());
}
