//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/RaggedRightIndexer.test.cc
//---------------------------------------------------------------------------//

#include "corecel/data/RaggedRightIndexer.hh"

#include "corecel/cont/Range.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

TEST(RaggedRightIndexerTest, basic)
{
    using RRI = RaggedRightIndexer<4>;
    using RRD = RaggedRightIndexerData<4>;
    using RC = RRI::Coords;

    RRD rrd({3, 4, 1, 2});
    RRI rri(rrd);

    // Flattened to ragged
    EXPECT_EQ(RC({0, 0}), rri.coords(0));
    EXPECT_EQ(RC({0, 1}), rri.coords(1));
    EXPECT_EQ(RC({0, 2}), rri.coords(2));
    EXPECT_EQ(RC({1, 0}), rri.coords(3));
    EXPECT_EQ(RC({1, 1}), rri.coords(4));
    EXPECT_EQ(RC({1, 2}), rri.coords(5));
    EXPECT_EQ(RC({1, 3}), rri.coords(6));
    EXPECT_EQ(RC({2, 0}), rri.coords(7));
    EXPECT_EQ(RC({3, 0}), rri.coords(8));
    EXPECT_EQ(RC({3, 1}), rri.coords(9));

    // Ragged to flattened
    EXPECT_EQ(0, rri.index(RC({0, 0})));
    EXPECT_EQ(1, rri.index(RC({0, 1})));
    EXPECT_EQ(2, rri.index(RC({0, 2})));
    EXPECT_EQ(3, rri.index(RC({1, 0})));
    EXPECT_EQ(4, rri.index(RC({1, 1})));
    EXPECT_EQ(5, rri.index(RC({1, 2})));
    EXPECT_EQ(6, rri.index(RC({1, 3})));
    EXPECT_EQ(7, rri.index(RC({2, 0})));
    EXPECT_EQ(8, rri.index(RC({3, 0})));
    EXPECT_EQ(9, rri.index(RC({3, 1})));
}

TEST(RaggedRightIndexerTest, TEST_IF_CELERITAS_DEBUG(error))
{
    EXPECT_THROW((RaggedRightIndexerData<3>({2, 0, 1})), DebugError);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
