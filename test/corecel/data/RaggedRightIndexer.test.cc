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

TEST(HyerslabIndexerTest, basic)
{
    using RRI = RaggedRightIndexer<size_type, 4>;
    using RI = RRI::RaggedIndices;

    RRI rri({3, 4, 1, 2});

    // Flattened to ragged
    EXPECT_EQ(RI({0, 0}), rri.ragged_indices(0));
    EXPECT_EQ(RI({0, 1}), rri.ragged_indices(1));
    EXPECT_EQ(RI({0, 2}), rri.ragged_indices(2));
    EXPECT_EQ(RI({1, 0}), rri.ragged_indices(3));
    EXPECT_EQ(RI({1, 1}), rri.ragged_indices(4));
    EXPECT_EQ(RI({1, 2}), rri.ragged_indices(5));
    EXPECT_EQ(RI({1, 3}), rri.ragged_indices(6));
    EXPECT_EQ(RI({2, 0}), rri.ragged_indices(7));
    EXPECT_EQ(RI({3, 0}), rri.ragged_indices(8));
    EXPECT_EQ(RI({3, 1}), rri.ragged_indices(9));

    // Ragged to flattened
    EXPECT_EQ(0, rri.flattened_index(RI({0, 0})));
    EXPECT_EQ(1, rri.flattened_index(RI({0, 1})));
    EXPECT_EQ(2, rri.flattened_index(RI({0, 2})));
    EXPECT_EQ(3, rri.flattened_index(RI({1, 0})));
    EXPECT_EQ(4, rri.flattened_index(RI({1, 1})));
    EXPECT_EQ(5, rri.flattened_index(RI({1, 2})));
    EXPECT_EQ(6, rri.flattened_index(RI({1, 3})));
    EXPECT_EQ(7, rri.flattened_index(RI({2, 0})));
    EXPECT_EQ(8, rri.flattened_index(RI({3, 0})));
    EXPECT_EQ(9, rri.flattened_index(RI({3, 1})));
}

#if CELERITAS_DEBUG
TEST(HyerslabIndexerTest, error)
{
    EXPECT_THROW((RaggedRightIndexer<size_type, 3>({2, 0, 3})), DebugError);
}
#endif

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
