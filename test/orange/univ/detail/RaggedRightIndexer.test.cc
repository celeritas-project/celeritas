//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/univ/detail/RaggedRightIndexer.test.cc
//---------------------------------------------------------------------------//
#include "orange/univ/detail/RaggedRightIndexer.hh"

#include "corecel/cont/Range.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace detail
{
namespace test
{
//---------------------------------------------------------------------------//

TEST(RaggedRightIndexerTest, basic)
{
    using RRD = RaggedRightIndexerData<4>;
    using Coords = Array<size_type, 2>;

    RRD const rrd = RRD::from_sizes({3, 4, 1, 2});
    RaggedRightIndexer<4> to_flat(rrd);
    RaggedRightInverseIndexer<4> from_flat(rrd);

    // Ragged to flattened
    EXPECT_EQ(0, to_flat(Coords({0, 0})));
    EXPECT_EQ(1, to_flat(Coords({0, 1})));
    EXPECT_EQ(2, to_flat(Coords({0, 2})));
    EXPECT_EQ(3, to_flat(Coords({1, 0})));
    EXPECT_EQ(4, to_flat(Coords({1, 1})));
    EXPECT_EQ(5, to_flat(Coords({1, 2})));
    EXPECT_EQ(6, to_flat(Coords({1, 3})));
    EXPECT_EQ(7, to_flat(Coords({2, 0})));
    EXPECT_EQ(8, to_flat(Coords({3, 0})));
    EXPECT_EQ(9, to_flat(Coords({3, 1})));

    // Flattened to ragged
    for (auto i : range(9))
    {
        EXPECT_EQ(to_flat(from_flat(i)), i);
    }
}

TEST(RaggedRightIndexerTest, TEST_IF_CELERITAS_DEBUG(error))
{
    EXPECT_THROW((RaggedRightIndexerData<3>::from_sizes({2, 0, 1})),
                 DebugError);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail
}  // namespace celeritas
