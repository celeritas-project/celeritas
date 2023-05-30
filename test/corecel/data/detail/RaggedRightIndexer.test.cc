//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/RaggedRightIndexer.test.cc
//---------------------------------------------------------------------------//

#include "corecel/data/detail/RaggedRightIndexer.hh"

#include "corecel/cont/Range.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

TEST(RaggedRightIndexerTest, basic)
{
    using RRD = RaggedRightIndexerData<4>;
    using Coords = Array<size_type, 2>;

    RRD rrd = RRD::from_sizes({3, 4, 1, 2});
    RaggedRightIndexer<4> rri(rrd);
    RaggedRightInverseIndexer<4> rrii(rrd);

    // Ragged to flattened
    EXPECT_EQ(0, rri(Coords({0, 0})));
    EXPECT_EQ(1, rri(Coords({0, 1})));
    EXPECT_EQ(2, rri(Coords({0, 2})));
    EXPECT_EQ(3, rri(Coords({1, 0})));
    EXPECT_EQ(4, rri(Coords({1, 1})));
    EXPECT_EQ(5, rri(Coords({1, 2})));
    EXPECT_EQ(6, rri(Coords({1, 3})));
    EXPECT_EQ(7, rri(Coords({2, 0})));
    EXPECT_EQ(8, rri(Coords({3, 0})));
    EXPECT_EQ(9, rri(Coords({3, 1})));

    // Flattened to ragged
    EXPECT_EQ(Coords({0, 0}), rrii(0));
    EXPECT_EQ(Coords({0, 1}), rrii(1));
    EXPECT_EQ(Coords({0, 2}), rrii(2));
    EXPECT_EQ(Coords({1, 0}), rrii(3));
    EXPECT_EQ(Coords({1, 1}), rrii(4));
    EXPECT_EQ(Coords({1, 2}), rrii(5));
    EXPECT_EQ(Coords({1, 3}), rrii(6));
    EXPECT_EQ(Coords({2, 0}), rrii(7));
    EXPECT_EQ(Coords({3, 0}), rrii(8));
    EXPECT_EQ(Coords({3, 1}), rrii(9));
}

TEST(RaggedRightIndexerTest, TEST_IF_CELERITAS_DEBUG(error))
{
    EXPECT_THROW((RaggedRightIndexerData<3>::from_sizes({2, 0, 1})),
                 DebugError);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
