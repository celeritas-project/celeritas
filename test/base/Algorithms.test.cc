//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Array.test.cc
//---------------------------------------------------------------------------//
#include "base/Algorithms.hh"

#include <algorithm>
#include <functional>
#include <type_traits>
#include <utility>
#include "celeritas_test.hh"

struct Foo
{
};

template<class Expected, class T>
void test_forward_impl(T&& val)
{
    EXPECT_TRUE((
        std::is_same<Expected, decltype(celeritas::cforward<T>(val))>::value));
}

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST(UtilityTest, forward)
{
    Foo       foo;
    const Foo cfoo;

    test_forward_impl<Foo&>(foo);
    test_forward_impl<const Foo&>(cfoo);
    test_forward_impl<Foo&&>(Foo{});
}

TEST(UtilityTest, move)
{
    Foo foo;

    EXPECT_TRUE((std::is_same<Foo&&, decltype(celeritas::cmove(foo))>::value));
    EXPECT_TRUE(
        (std::is_same<Foo&&, decltype(celeritas::cmove(Foo{}))>::value));
}

TEST(UtilityTest, swap)
{
    using celeritas::cswap;

    // Test trivial type swapping
    {
        int a = 1;
        int b = 2;
        cswap(a, b);
        EXPECT_EQ(2, a);
        EXPECT_EQ(1, b);
    }

    {
        std::string a{"foo"};
        std::string b{"barbarbarbarbar"};
        cswap(a, b);
        EXPECT_EQ("barbarbarbarbar", a);
        EXPECT_EQ("foo", b);
    }
}

//---------------------------------------------------------------------------//

TEST(AlgorithmsTest, clamp)
{
    using celeritas::clamp_to_nonneg;
    constexpr auto nan = std::numeric_limits<double>::quiet_NaN();

    EXPECT_DOUBLE_EQ(1.2345, clamp_to_nonneg(1.2345));
    EXPECT_DOUBLE_EQ(0.0, clamp_to_nonneg(-123));
    EXPECT_TRUE(std::isnan(celeritas::clamp_to_nonneg(nan)));
}

TEST(AlgorithmsTest, lower_bound)
{
    // Test empty vector
    std::vector<int> v;
    EXPECT_EQ(0, celeritas::lower_bound(v.begin(), v.end(), 10) - v.begin());

    // Test a selection of sorted values, and values surroundig them
    v = {-3, 1, 4, 9, 10, 11, 15, 15};

    for (int val : v)
    {
        for (int delta : {-1, 0, 1})
        {
            auto expected = std::lower_bound(v.begin(), v.end(), val + delta);
            auto actual
                = celeritas::lower_bound(v.begin(), v.end(), val + delta);
            EXPECT_EQ(expected - v.begin(), actual - v.begin())
                << "Lower bound failed for value " << val + delta;
        }
    }
}

TEST(AlgorithmsTest, minmax)
{
    EXPECT_EQ(1, celeritas::min<int>(1, 2));
    EXPECT_EQ(2, celeritas::max<int>(1, 2));
}

TEST(AlgorithmsTest, min_element)
{
    std::vector<int> v;

    auto             min_element_idx = [&v]() {
        return celeritas::min_element(v.begin(), v.end()) - v.begin();
    };
    auto min_element_gt_idx = [&v]() {
        return celeritas::min_element(v.begin(), v.end(), std::greater<int>())
               - v.begin();
    };

    // Empty vector will return 0, which is off-the-end
    EXPECT_EQ(0, min_element_idx());
    EXPECT_EQ(0, min_element_gt_idx());

    v = {100};
    EXPECT_EQ(0, min_element_idx());
    EXPECT_EQ(0, min_element_gt_idx());

    v = {10, 2, 100, 3, -1};
    EXPECT_EQ(4, min_element_idx());
    EXPECT_EQ(2, min_element_gt_idx());

    v[2] = -100;
    EXPECT_EQ(2, min_element_idx());
    EXPECT_EQ(0, min_element_gt_idx());
}

//---------------------------------------------------------------------------//

TEST(MathTest, ipow)
{
    EXPECT_DOUBLE_EQ(1, celeritas::ipow<0>(0.0));
    EXPECT_EQ(123.456, celeritas::ipow<1>(123.456));
    EXPECT_EQ(8, (celeritas::ipow<3>(2)));
    EXPECT_FLOAT_EQ(0.001f, celeritas::ipow<3>(0.1f));
    EXPECT_EQ(1e4, celeritas::ipow<4>(10.0));
    EXPECT_TRUE((std::is_same<int, decltype(celeritas::ipow<4>(5))>::value));
}
