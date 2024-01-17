//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
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
    Array<size_type, 2> const dims{3, 4};
    HyperslabIndexer<2> to_index(dims);
    HyperslabInverseIndexer<2> to_coords(dims);

    size_type index = 0;
    for (size_type a : range(3))
    {
        for (size_type b : range(4))
        {
            Array<size_type, 2> coords{a, b};
            EXPECT_EQ(index, to_index(coords));
            EXPECT_EQ(coords, to_coords(index));
            index++;
        }
    }
}

TEST(HyperslabIndexerTest, 3D)
{
    Array<size_type, 3> const dims{3, 4, 5};
    HyperslabIndexer<3> to_index(dims);
    HyperslabInverseIndexer<3> to_coords(dims);

    size_type index = 0;
    for (size_type a : range(3))
    {
        for (size_type b : range(4))
        {
            for (size_type c : range(5))
            {
                Array<size_type, 3> coords{a, b, c};
                EXPECT_EQ(index, to_index(coords));
                EXPECT_EQ(coords, to_coords(index));
                index++;
            }
        }
    }
}

TEST(HyperslabIndexerTest, 4D)
{
    Array<size_type, 4> const dims{4, 6, 3, 2};
    HyperslabIndexer<4> to_index(dims);
    HyperslabInverseIndexer<4> to_coords(dims);

    size_type index = 0;
    for (size_type a : range(4))
    {
        for (size_type b : range(6))
        {
            for (size_type c : range(3))
            {
                for (size_type d : range(2))
                {
                    Array<size_type, 4> coords{a, b, c, d};
                    EXPECT_EQ(index, to_index(coords));
                    EXPECT_EQ(coords, to_coords(index));
                    index++;
                }
            }
        }
    }
}

TEST(HyperslabIndexerTest, 5D_with_ones)
{
    Array<size_type, 5> const dims{3, 1, 4, 1, 5};
    HyperslabIndexer<5> to_index(dims);
    HyperslabInverseIndexer<5> to_coords(dims);

    size_type index = 0;
    for (size_type a : range(3))
    {
        for (size_type b : range(4))
        {
            for (size_type c : range(5))
            {
                Array<size_type, 5> coords{a, 0, b, 0, c};
                EXPECT_EQ(index, to_index(coords));
                EXPECT_EQ(coords, to_coords(index));
                index++;
            }
        }
    }
}

TEST(HyperslabIndexerTest, TEST_IF_CELERITAS_DEBUG(error))
{
    Array<size_type, 3> const dims{2, 0, 3};
    EXPECT_THROW((HyperslabIndexer<3>(dims)), DebugError);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
