//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/cont/Array.test.cc
//---------------------------------------------------------------------------//
#include "corecel/cont/Array.hh"

#include <type_traits>

#include "corecel/cont/EnumArray.hh"

#include "TestMacros.hh"
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

TEST(ArrayTest, single_level)
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

TEST(ArrayTest, deduction)
{
    Array<double, 3> y = {1, 3.0, 2.0f};
    EXPECT_TRUE((std::is_same_v<decltype(y), Array<double, 3>>));

    static double const expected_y[] = {1.0, 3.0, 2.0};
    EXPECT_VEC_EQ(expected_y, y);
}

TEST(ArrayTest, two_level)
{
    using Int3 = Array<int, 3>;
    Array<Int3, 3> x = {{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}};
    EXPECT_VEC_EQ((Int3{1, 2, 3}), x[0]);
    EXPECT_VEC_EQ((Int3{4, 5, 6}), x[1]);
    EXPECT_VEC_EQ((Int3{7, 8, 9}), x[2]);
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
