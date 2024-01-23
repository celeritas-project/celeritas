//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/ArrayOperators.test.cc
//---------------------------------------------------------------------------//
#include "corecel/math/ArrayOperators.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

TEST(ArrayOperatorsTest, add)
{
    using Int2 = Array<int, 2>;
    Int2 arr{1, 2};
    {
        auto const& result = (arr += Int2{2, 3});
        EXPECT_EQ(&result, &arr);
    }
    EXPECT_EQ((Int2{3, 5}), arr);

    {
        auto const& result = (arr += 4);
        EXPECT_EQ(&result, &arr);
    }
    EXPECT_EQ((Int2{7, 9}), arr);

    EXPECT_EQ((Int2{3, 4}), (Int2{1, 2} + Int2{2, 2}));
    EXPECT_EQ((Int2{3, 4}), (Int2{1, 2} + 2));
}

TEST(ArrayOperatorsTest, sub)
{
    using Int2 = Array<int, 2>;
    Int2 arr{1, 2};
    arr -= Int2{2, 3};
    EXPECT_EQ((Int2{-1, -1}), arr);

    arr -= 4;
    EXPECT_EQ((Int2{-5, -5}), arr);

    EXPECT_EQ((Int2{3, 4}), (Int2{5, 6} - Int2{2, 2}));
    EXPECT_EQ((Int2{3, 4}), (Int2{5, 6} - 2));
}

TEST(ArrayOperatorsTest, mul)
{
    using Int2 = Array<int, 2>;
    Int2 arr{1, 2};
    arr *= Int2{2, 3};
    EXPECT_EQ((Int2{2, 6}), arr);

    arr *= 4;
    EXPECT_EQ((Int2{8, 24}), arr);

    EXPECT_EQ((Int2{10, 12}), (Int2{5, 6} * Int2{2, 2}));
    EXPECT_EQ((Int2{10, 12}), (Int2{5, 6} * 2));

    // Multiplication only: left-multiply by scalar
    EXPECT_EQ((Int2{10, 12}), (2 * Int2{5, 6}));
}

TEST(ArrayOperatorsTest, div)
{
    using Int2 = Array<int, 2>;
    Int2 arr{4, 6};
    arr /= Int2{2, 3};
    EXPECT_EQ((Int2{2, 2}), arr);

    arr /= 2;
    EXPECT_EQ((Int2{1, 1}), arr);

    EXPECT_EQ((Int2{3, 4}), (Int2{6, 8} / Int2{2, 2}));
    EXPECT_EQ((Int2{3, 4}), (Int2{6, 8} / 2));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
