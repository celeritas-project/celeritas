//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/cont/Array.test.cc
//---------------------------------------------------------------------------//
#include "corecel/cont/Array.hh"

#include "corecel/cont/EnumArray.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

enum class Color : unsigned int
{
    red,
    green,
    blue,
    size_
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST(ArrayTest, all)
{
    // Note: C++14 required to write without double brackets
    Array<int, 3> x = {1, 3, 2};

    EXPECT_FALSE(x.empty());
    EXPECT_EQ(3, x.size());
    EXPECT_EQ(1, x.front());
    EXPECT_EQ(2, x.back());
    EXPECT_EQ(3, x[1]);
    EXPECT_EQ(static_cast<void*>(&x), x.data());

    auto const& cx = x;
    EXPECT_FALSE(cx.empty());
    EXPECT_EQ(3, cx.size());
    EXPECT_EQ(1, cx.front());
    EXPECT_EQ(2, cx.back());
    EXPECT_EQ(3, cx[1]);
    EXPECT_EQ(static_cast<void const*>(&x), cx.data());

    Array<int, 3> y{20, 30, 40};
    EXPECT_EQ(x, x);
    EXPECT_NE(x, y);

    y = x;
    EXPECT_EQ(y, x);

    y.fill(4);
    EXPECT_EQ(4, y.front());
    EXPECT_EQ(4, y.back());

    for (int& v : y)
    {
        v = 3;
    }
}

TEST(EnumArrayTest, all)
{
    EnumArray<Color, int> x = {1, 3, 2};
    EXPECT_EQ(3, x.size());

    EXPECT_FALSE(x.empty());
    EXPECT_EQ(3, x.size());
    EXPECT_EQ(1, x.front());
    EXPECT_EQ(2, x.back());
    EXPECT_EQ(1, x[Color::red]);
    EXPECT_EQ(3, x[Color::green]);
    EXPECT_EQ(static_cast<void*>(&x), x.data());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
