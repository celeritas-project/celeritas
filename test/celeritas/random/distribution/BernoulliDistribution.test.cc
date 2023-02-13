//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/random/distribution/BernoulliDistribution.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/random/distribution/BernoulliDistribution.hh"

#include <random>

#include "corecel/cont/Range.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

TEST(BernoulliDistributionTest, single_constructor)
{
    std::mt19937 rng;
    BernoulliDistribution quarter_true(0.25);
    EXPECT_SOFT_EQ(0.25, quarter_true.p());

    int num_true = 0;
    for ([[maybe_unused]] auto i : range(1000))
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

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
