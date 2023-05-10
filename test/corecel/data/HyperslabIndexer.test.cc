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
    HyperslabIndexer<size_type, 2> hi({3, 4});

    size_type index = 0;
    for (size_type i : range(3))
    {
        for (size_type j : range(4))
        {
            Array<size_type, 2> coords{i, j};
            EXPECT_EQ(coords, hi.coords(index));
            EXPECT_EQ(index, hi.index(coords));
            index++;
        }
    }
}

TEST(HyperslabIndexerTest, 3D)
{
    HyperslabIndexer<size_type, 3> hi({3, 4, 5});

    size_type index = 0;
    for (size_type i : range(3))
    {
        for (size_type j : range(4))
        {
            for (size_type k : range(5))
            {
                Array<size_type, 3> coords{i, j, k};
                EXPECT_EQ(coords, hi.coords(index));
                EXPECT_EQ(index, hi.index(coords));
                index++;
            }
        }
    }
}

TEST(HyperslabIndexerTest, 4D)
{
    HyperslabIndexer<size_type, 4> hi({4, 6, 3, 2});

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
                    EXPECT_EQ(coords, hi.coords(index));
                    EXPECT_EQ(index, hi.index(coords));
                    index++;
                }
            }
        }
    }
}

TEST(HyperslabIndexerTest, 5D_with_ones)
{
    HyperslabIndexer<size_type, 5> hi({3, 1, 4, 1, 5});

    size_type index = 0;
    for (size_type i : range(3))
    {
        for (size_type j : range(4))
        {
            for (size_type k : range(5))
            {
                Array<size_type, 5> coords{i, 0, j, 0, k};
                EXPECT_EQ(coords, hi.coords(index));
                EXPECT_EQ(index, hi.index(coords));
                index++;
            }
        }
    }
}

#if CELERITAS_DEBUG
TEST(HyperslabIndexerTest, error)
{
    EXPECT_THROW((HyperslabIndexer<size_type, 3>({2, 0, 3})), DebugError);
}
#endif

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
