//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/HyperslabIndexer.test.cc
//---------------------------------------------------------------------------//

#include "corecel/data/HyperslabIndexer.hh"

#include "corecel/cont/Range.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

TEST(HyperslabIndexerTest, 2D)
{
    HyperslabIndexer<2> hi({3, 4});
    HyperslabInverseIndexer<2> hii({3, 4});

    size_type index = 0;
    for (size_type i : range(3))
    {
        for (size_type j : range(4))
        {
            Array<size_type, 2> coords{i, j};
            EXPECT_EQ(index, hi(coords));
            EXPECT_EQ(coords, hii(index));
            index++;
        }
    }
}

TEST(HyperslabIndexerTest, 3D)
{
    HyperslabIndexer<3> hi({3, 4, 5});
    HyperslabInverseIndexer<3> hii({3, 4, 5});

    size_type index = 0;
    for (size_type i : range(3))
    {
        for (size_type j : range(4))
        {
            for (size_type k : range(5))
            {
                Array<size_type, 3> coords{i, j, k};
                EXPECT_EQ(index, hi(coords));
                EXPECT_EQ(coords, hii(index));
                index++;
            }
        }
    }
}

TEST(HyperslabIndexerTest, 4D)
{
    HyperslabIndexer<4> hi({4, 6, 3, 2});
    HyperslabInverseIndexer<4> hii({4, 6, 3, 2});

    size_type index = 0;
    for (size_type i : range(4))
    {
        for (size_type j : range(6))
        {
            for (size_type k : range(3))
            {
                for (size_type l : range(2))
                {
                    Array<size_type, 4> coords{i, j, k, l};
                    EXPECT_EQ(index, hi(coords));
                    EXPECT_EQ(coords, hii(index));
                    index++;
                }
            }
        }
    }
}

TEST(HyperslabIndexerTest, 5D_with_ones)
{
    HyperslabIndexer<5> hi({3, 1, 4, 1, 5});
    HyperslabInverseIndexer<5> hii({3, 1, 4, 1, 5});

    size_type index = 0;
    for (size_type i : range(3))
    {
        for (size_type j : range(4))
        {
            for (size_type k : range(5))
            {
                Array<size_type, 5> coords{i, 0, j, 0, k};
                EXPECT_EQ(index, hi(coords));
                EXPECT_EQ(coords, hii(index));
                index++;
            }
        }
    }
}

TEST(HyperslabIndexerTest, TEST_IF_CELERITAS_DEBUG(error))
{
    EXPECT_THROW((HyperslabIndexer<3>({2, 0, 3})), DebugError);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
