//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/Algorithms.test.cc
//---------------------------------------------------------------------------//
#include "corecel/math/Algorithms.hh"

#include <algorithm>
#include <functional>
#include <type_traits>
#include <utility>

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

struct Foo
{
};

template<class Expected, class T>
void test_forward_impl(T&& val)
{
    EXPECT_TRUE((
        std::is_same<Expected, decltype(::celeritas::forward<T>(val))>::value));
}

struct IsInRange
{
    int start;
    int stop;

    bool operator()(int value) const { return value >= start && value < stop; }
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST(UtilityTest, forward)
{
    Foo foo;
    Foo const cfoo;

    test_forward_impl<Foo&>(foo);
    test_forward_impl<Foo const&>(cfoo);
    test_forward_impl<Foo&&>(Foo{});
}

TEST(UtilityTest, move)
{
    Foo foo;

    EXPECT_TRUE((std::is_same<Foo&&, decltype(::celeritas::move(foo))>::value));
    EXPECT_TRUE(
        (std::is_same<Foo&&, decltype(::celeritas::move(Foo{}))>::value));
}

TEST(UtilityTest, trivial_swap)
{
    int a = 1;
    int b = 2;
    trivial_swap(a, b);
    EXPECT_EQ(2, a);
    EXPECT_EQ(1, b);
}

TEST(UtilityTest, exchange)
{
    int dst = 456;
    EXPECT_EQ(456, exchange(dst, 123));
    EXPECT_EQ(123, dst);
}

//---------------------------------------------------------------------------//

TEST(AlgorithmsTest, all_of)
{
    static bool const items[] = {true, false, true, true};
    auto is_true = [](bool b) { return b; };
    EXPECT_TRUE(all_of(std::begin(items), std::begin(items), is_true));
    EXPECT_FALSE(all_of(std::begin(items), std::end(items), is_true));
    EXPECT_TRUE(all_of(std::begin(items) + 2, std::end(items), is_true));
}

TEST(AlgorithmsTest, any_of)
{
    static bool const items[] = {false, true, false, false};
    auto is_true = [](bool b) { return b; };
    EXPECT_FALSE(any_of(std::begin(items), std::begin(items), is_true));
    EXPECT_TRUE(any_of(std::begin(items), std::end(items), is_true));
    EXPECT_FALSE(any_of(std::begin(items) + 2, std::end(items), is_true));
}

TEST(AlgorithmsTest, clamp)
{
    EXPECT_EQ(123, clamp(123, 100, 200));
    EXPECT_EQ(100, clamp(99, 100, 200));
    EXPECT_EQ(200, clamp(999, 100, 200));
    if (CELERITAS_DEBUG)
    {
        EXPECT_THROW(clamp(150, 200, 100), DebugError);
    }
}

TEST(AlgorithmsTest, clamp_to_nonneg)
{
    constexpr auto nan = std::numeric_limits<double>::quiet_NaN();

    EXPECT_DOUBLE_EQ(1.2345, clamp_to_nonneg(1.2345));
    EXPECT_DOUBLE_EQ(0.0, clamp_to_nonneg(-123));
    EXPECT_TRUE(std::isnan(clamp_to_nonneg(nan)));
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

TEST(AlgorithmsTest, lower_bound_linear)
{
    // Test empty vector
    std::vector<int> v;
    EXPECT_EQ(
        0, celeritas::lower_bound_linear(v.begin(), v.end(), 10) - v.begin());

    // Test a selection of sorted values, and values surroundig them
    v = {-3, 1, 4, 9, 10, 11, 15, 15};

    for (int val : v)
    {
        for (int delta : {-1, 0, 1})
        {
            auto expected = std::lower_bound(v.begin(), v.end(), val + delta);
            auto actual = celeritas::lower_bound_linear(
                v.begin(), v.end(), val + delta);
            EXPECT_EQ(expected - v.begin(), actual - v.begin())
                << "Lower bound failed for value " << val + delta;
        }
    }
}

TEST(AlgorithmsTest, upper_bound)
{
    // Test empty vector
    std::vector<int> v;
    EXPECT_EQ(0, celeritas::upper_bound(v.begin(), v.end(), 10) - v.begin());

    // Test a selection of sorted values, and values surrounding them
    v = {-3, 1, 4, 9, 10, 11, 15, 15};

    for (int val : v)
    {
        for (int delta : {-1, 0, 1})
        {
            auto expected = std::upper_bound(v.begin(), v.end(), val + delta);
            auto actual
                = celeritas::upper_bound(v.begin(), v.end(), val + delta);
            EXPECT_EQ(expected - v.begin(), actual - v.begin())
                << "Upper bound failed for value " << val + delta;
        }
    }
}

TEST(AlgorithmsTest, partition)
{
    std::vector<int> values{-1, 2, 3, 4, 2, 6, 9, 4};
    celeritas::partition(values.begin(), values.end(), IsInRange{2, 4});

    static int const expected_values[] = {2, 2, 3, 4, -1, 6, 9, 4};
    EXPECT_VEC_EQ(expected_values, values);
}

TEST(AlgorithmsTest, sort)
{
    std::vector<int> data;
    {
        celeritas::sort(data.begin(), data.end());
        EXPECT_EQ(0, data.size());
    }
    {
        data = {123};
        celeritas::sort(data.begin(), data.end());
        EXPECT_EQ(123, data.front());
    }
    {
        data = {1, 2, 4, 3, -1, 123, 2};
        celeritas::sort(data.begin(), data.end());
        static int const expected_data[] = {-1, 1, 2, 2, 3, 4, 123};
        EXPECT_VEC_EQ(expected_data, data);
    }
    {
        data = {1, 2, 4, 3, -1, 123, 2};
        celeritas::sort(data.begin(), data.end(), std::greater<>{});
        static int const expected_data[] = {123, 4, 3, 2, 2, 1, -1};
        EXPECT_VEC_EQ(expected_data, data);
    }
}
TEST(AlgorithmsTest, minmax)
{
    EXPECT_EQ(1, min<int>(1, 2));
    EXPECT_EQ(2, max<int>(1, 2));
}

TEST(AlgorithmsTest, min_element)
{
    std::vector<int> v;

    auto min_element_idx = [&v]() {
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
    EXPECT_DOUBLE_EQ(1, ipow<0>(0.0));
    EXPECT_EQ(123.456, ipow<1>(123.456));
    EXPECT_EQ(8, ipow<3>(2));
    EXPECT_FLOAT_EQ(0.001f, ipow<3>(0.1f));
    EXPECT_EQ(1e4, ipow<4>(10.0));
    EXPECT_TRUE((std::is_same<int, decltype(ipow<4>(5))>::value));
}

//---------------------------------------------------------------------------//

TEST(MathTest, fastpow)
{
    EXPECT_DOUBLE_EQ(0.0, fastpow(0.0, 1.0));
    EXPECT_DOUBLE_EQ(0.0, fastpow(0.0, 5.55042));
    EXPECT_DOUBLE_EQ(1.0, fastpow(1234.0, 0.0));
    if (CELERITAS_DEBUG)
    {
        EXPECT_THROW(fastpow(0.0, 0.0), DebugError);
    }
    EXPECT_DOUBLE_EQ(123.456, fastpow(123.456, 1.0));
    EXPECT_FLOAT_EQ(0.001f, fastpow(0.1f, 3.0f));
    EXPECT_DOUBLE_EQ(10.0, fastpow(1000.0, 1.0 / 3.0));
    EXPECT_DOUBLE_EQ(1.0 / 32.0, fastpow(2.0, -5.0));

    EXPECT_TRUE((std::is_same<float, decltype(fastpow(5.0f, 1.0f))>::value));
}

//---------------------------------------------------------------------------//

TEST(MathTest, rsqrt)
{
    constexpr auto dblinf = std::numeric_limits<double>::infinity();
    EXPECT_DOUBLE_EQ(0.5, rsqrt(4.0));
    EXPECT_DOUBLE_EQ(dblinf, rsqrt(0.0));
    EXPECT_DOUBLE_EQ(0.0, rsqrt(dblinf));

    constexpr auto fltinf = std::numeric_limits<float>::infinity();
    EXPECT_FLOAT_EQ(0.5f, rsqrt(4.0f));
    EXPECT_FLOAT_EQ(fltinf, rsqrt(0.0f));
    EXPECT_FLOAT_EQ(0.0f, rsqrt(fltinf));
}

//---------------------------------------------------------------------------//

TEST(MathTest, ceil_div)
{
    EXPECT_EQ(0u, ceil_div(0u, 32u));
    EXPECT_EQ(1u, ceil_div(1u, 32u));
    EXPECT_EQ(1u, ceil_div(32u, 32u));
    EXPECT_EQ(2u, ceil_div(33u, 32u));
    EXPECT_EQ(8u, ceil_div(50u, 7u));
}

//---------------------------------------------------------------------------//

TEST(MathTest, negate)
{
    double const zero = 0;
    auto negzero = -zero;
    EXPECT_TRUE(std::signbit(negzero));
    EXPECT_FALSE(std::signbit(negate(zero)));

    constexpr auto dblinf = std::numeric_limits<double>::infinity();
    EXPECT_DOUBLE_EQ(-2.0, negate(2.0));
    EXPECT_DOUBLE_EQ(-dblinf, negate(dblinf));
    EXPECT_TRUE(std::isnan(negate(std::numeric_limits<double>::quiet_NaN())));
}

//---------------------------------------------------------------------------//

TEST(MathTest, diffsq)
{
    EXPECT_DOUBLE_EQ(9.0, diffsq(5.0, 4.0));
    EXPECT_DOUBLE_EQ(ipow<2>(std::sin(0.2)), diffsq(1.0, std::cos(0.2)));

    float a{10000.001}, b{10000}, actual{20};
    EXPECT_FLOAT_EQ(0.46875f, actual - diffsq(a, b));
    EXPECT_LE(actual - diffsq(a, b), actual - (a * a - b * b));
}

//---------------------------------------------------------------------------//

TEST(MathTest, sincos)
{
    {
        double s{0}, c{0};
        sincos(0.123, &s, &c);
        EXPECT_DOUBLE_EQ(std::sin(0.123), s);
        EXPECT_DOUBLE_EQ(std::cos(0.123), c);
    }
    {
        float s{0}, c{0};
        sincos(0.123f, &s, &c);
        EXPECT_FLOAT_EQ(std::sin(0.123f), s);
        EXPECT_FLOAT_EQ(std::cos(0.123f), c);
    }
}

//---------------------------------------------------------------------------//

TEST(MathTest, sincospi)
{
    EXPECT_DOUBLE_EQ(std::sin(m_pi * 0.1), sinpi(0.1));
    EXPECT_DOUBLE_EQ(std::cos(m_pi * 0.1), cospi(0.1));

    double s{0}, c{0};
    sincospi(0.123, &s, &c);
    EXPECT_DOUBLE_EQ(std::sin(m_pi * 0.123), s);
    EXPECT_DOUBLE_EQ(std::cos(m_pi * 0.123), c);

    // Test special cases
    sincospi(0, &s, &c);
    EXPECT_EQ(double(0.0), s);
    EXPECT_EQ(double(1.0), c);

    sincospi(0.5, &s, &c);
    EXPECT_EQ(double(1.0), s);
    EXPECT_EQ(double(0.0), c);

    sincospi(1.0, &s, &c);
    EXPECT_EQ(double(0.0), s);
    EXPECT_EQ(double(-1.0), c);

    sincospi(1.5, &s, &c);
    EXPECT_EQ(double(-1.0), s);
    EXPECT_EQ(double(0.0), c);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
