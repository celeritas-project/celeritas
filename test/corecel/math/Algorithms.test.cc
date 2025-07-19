//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/Algorithms.test.cc
//---------------------------------------------------------------------------//
#include "corecel/math/Algorithms.hh"

#include <algorithm>
#include <functional>
#include <type_traits>
#include <utility>

#include "corecel/Constants.hh"
#include "corecel/data/CollectionMirror.hh"

#include "Algorithms.test.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
using constants::pi;

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
    LogicalTrue<bool> is_true;
    EXPECT_TRUE(all_of(std::begin(items), std::begin(items), is_true));
    EXPECT_FALSE(all_of(std::begin(items), std::end(items), is_true));
    EXPECT_TRUE(all_of(std::begin(items) + 2, std::end(items), is_true));

    LogicalFalse<bool> is_false;
    EXPECT_FALSE(all_of(std::begin(items), std::end(items), is_false));
    EXPECT_TRUE(all_of(std::begin(items) + 1, std::begin(items) + 2, is_false));
}

TEST(AlgorithmsTest, any_of)
{
    static bool const items[] = {false, true, false, false};
    LogicalTrue<> is_true;
    EXPECT_FALSE(any_of(std::begin(items), std::begin(items), is_true));
    EXPECT_TRUE(any_of(std::begin(items), std::end(items), is_true));
    EXPECT_FALSE(any_of(std::begin(items) + 2, std::end(items), is_true));
}

TEST(AlgorithmsTest, all_adjacent)
{
    static int const incr[] = {0, 1, 3, 20, 200};
    static int const vee[] = {3, 2, 1, 2, 3};
    static int const nondecr[] = {1, 1, 2, 3, 5, 8};

    // Empty and single-element ranges
    EXPECT_TRUE(all_adjacent(
        std::begin(incr), std::begin(incr), [](int, int) { return false; }));
    EXPECT_TRUE(all_adjacent(std::begin(incr),
                             std::begin(incr) + 1,
                             [](int, int) { return false; }));

    auto lt = [](int a, int b) { return a < b; };
    auto le = [](int a, int b) { return a <= b; };
    EXPECT_TRUE(all_adjacent(std::begin(incr), std::end(incr), lt));
    EXPECT_FALSE(all_adjacent(std::begin(vee), std::end(vee), lt));
    EXPECT_FALSE(all_adjacent(std::begin(nondecr), std::end(nondecr), lt));

    EXPECT_TRUE(all_adjacent(std::begin(incr), std::end(incr), le));
    EXPECT_FALSE(all_adjacent(std::begin(vee), std::end(vee), le));
    EXPECT_TRUE(all_adjacent(std::begin(nondecr), std::end(nondecr), le));
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

    constexpr auto nan = std::numeric_limits<real_type>::quiet_NaN();
    EXPECT_TRUE(std::isnan(clamp(nan, real_type{-1}, real_type{1})));
}

TEST(AlgorithmsTest, clamp_to_nonneg)
{
    EXPECT_DOUBLE_EQ(1.2345, clamp_to_nonneg(1.2345));
    EXPECT_DOUBLE_EQ(0.0, clamp_to_nonneg(-123));
    EXPECT_EQ(pi, clamp_to_nonneg(pi));

    constexpr auto nan = std::numeric_limits<double>::quiet_NaN();
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

TEST(AlgorithmsTest, find_sorted)
{
    std::vector<int> v;
    auto find_index = [&v](int value) -> int {
        auto iter = celeritas::find_sorted(v.begin(), v.end(), value);
        if (iter == v.end())
            return -1;
        return iter - v.begin();
    };

    // Test empty vector
    EXPECT_EQ(-1, find_index(10));

    // Test a selection of sorted values
    v = {-3, 1, 4, 9, 10, 11, 15, 15};
    EXPECT_EQ(-1, find_index(-5));
    EXPECT_EQ(0, find_index(-3));
    EXPECT_EQ(2, find_index(4));
    EXPECT_EQ(-1, find_index(5));
    EXPECT_EQ(6, find_index(15));
    EXPECT_EQ(-1, find_index(16));
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

    constexpr auto nan = std::numeric_limits<real_type>::quiet_NaN();
    EXPECT_EQ(real_type{1}, min(real_type{1}, nan));
    EXPECT_EQ(real_type{1}, min(nan, real_type{1}));
    EXPECT_EQ(real_type{1}, max(real_type{1}, nan));
    EXPECT_EQ(real_type{1}, max(nan, real_type{1}));
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

    EXPECT_EQ(pi * pi, static_cast<double>(ipow<2>(pi)));
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

TEST(MathTest, fma)
{
    EXPECT_DOUBLE_EQ(std::fma(1.0, 2.0, 8.0), fma(1.0, 2.0, 8.0));

    EXPECT_DOUBLE_EQ(1 * 2 + 8, fma(1, 2, 8));
}

//---------------------------------------------------------------------------//

TEST(MathTest, hypot)
{
    static double const nums[] = {
        1.1e-10,
        0.456e-7,
        0.301e-5,
        0.6789e-3,
        0.1,
        3.123,
        -0.0,
        0.0,
    };
    for (double a : nums)
    {
        for (double b : nums)
        {
            for (auto i : range(1 << 4))
            {
                // Do combinations of flipped signs and inverted
                if (i & (1 << 0))
                    a = -a;
                if (i & (1 << 1))
                    b = -b;
                if (i & (1 << 2))
                    a = 1 / a;
                if (i & (1 << 3))
                    b = 1 / b;

                EXPECT_DOUBLE_EQ(std::hypot(a, b), hypot(a, b))
                    << "a=" << repr(a) << ", b=" << repr(b);

                auto af = static_cast<float>(a);
                auto bf = static_cast<float>(b);
                if (false)
                {
                    // The current implementation is not symmetric
                    EXPECT_EQ(hypot(af, bf), hypot(bf, af))
                        << "af=" << repr(af) << ", bf=" << repr(bf)
                        << ", exact: "
                        << repr(static_cast<float>(
                               std::hypot(static_cast<double>(af),
                                          static_cast<double>(bf))));
                }
                EXPECT_FLOAT_EQ(std::hypot(af, bf), hypot(af, bf))
                    << "af=" << repr(af) << ", bf=" << repr(bf);
            }
        }
    }
    EXPECT_DOUBLE_EQ(5.0, hypot(3.0, 4.0));
    EXPECT_FLOAT_EQ(5.0f, hypot(3.0f, 4.0f));
}

TEST(MathTest, hypot3)
{
    EXPECT_DOUBLE_EQ(std::hypot(1.0, 2.0, 3.0), hypot(1.0, 2.0, 3.0));
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

TEST(MathTest, local_work_calculator)
{
    {
        LocalWorkCalculator<unsigned int> calc_local_work{12, 4};
        for (auto i : range(4))
        {
            EXPECT_EQ(3, calc_local_work(i));
        }
        if (CELERITAS_DEBUG)
        {
            EXPECT_THROW(calc_local_work(4), DebugError);
        }
    }
    {
        LocalWorkCalculator<unsigned int> calc_local_work{7, 5};
        EXPECT_EQ(2, calc_local_work(0));
        EXPECT_EQ(2, calc_local_work(1));
        EXPECT_EQ(1, calc_local_work(2));
        EXPECT_EQ(1, calc_local_work(3));
        EXPECT_EQ(1, calc_local_work(4));
    }
    {
        LocalWorkCalculator<unsigned int> calc_local_work{2, 4};
        EXPECT_EQ(1, calc_local_work(0));
        EXPECT_EQ(1, calc_local_work(1));
        EXPECT_EQ(0, calc_local_work(2));
        EXPECT_EQ(0, calc_local_work(3));
    }
    {
        LocalWorkCalculator<unsigned int> calc_local_work{0, 1};
        EXPECT_EQ(0, calc_local_work(0));
    }
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

    float a{10000.001f}, b{10000.f}, actual{20.f};
    EXPECT_FLOAT_EQ(0.46875f, actual - diffsq(a, b));
    EXPECT_LE(actual - diffsq(a, b), actual - (a * a - b * b));
}

//---------------------------------------------------------------------------//

TEST(MathTest, eumod)
{
    // Wrap numbers to between [0, 360)
    EXPECT_DOUBLE_EQ(270.0, eumod(-90.0 - 360.0, 360.0));
    EXPECT_DOUBLE_EQ(270.0, eumod(-90.0, 360.0));
    EXPECT_DOUBLE_EQ(0.0, eumod(0.0, 360.0));
    EXPECT_DOUBLE_EQ(45.0, eumod(45.0, 360.0));
    EXPECT_DOUBLE_EQ(0.0, eumod(360.0, 360.0));
    EXPECT_DOUBLE_EQ(15.0, eumod(375.0, 360.0));
    EXPECT_DOUBLE_EQ(30.0, eumod(720.0 + 30, 360.0));

    // Edge case where the result can equal the denominator due to FP precision
    constexpr double eps = 1e-13;
    EXPECT_DOUBLE_EQ(360.0, eumod(-eps, 360.0));
    EXPECT_DOUBLE_EQ(360.0, eumod(-eps, -360.0));

    EXPECT_DOUBLE_EQ(eps, eumod(eps, 360.0));
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
    EXPECT_DOUBLE_EQ(std::sin(pi * 0.1), sinpi(0.1));
    EXPECT_DOUBLE_EQ(std::cos(pi * 0.1), cospi(0.1));

    {
        double s{0}, c{0};
        sincospi(0.123, &s, &c);
        EXPECT_DOUBLE_EQ(std::sin(pi * 0.123), s);
        EXPECT_DOUBLE_EQ(std::cos(pi * 0.123), c);

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
    {
        // This is about the threshold where sin(x) == 1.0f
        float inp{0.000233115f};
        float s{0}, c{0};
        sincospi(inp, &s, &c);
        EXPECT_FLOAT_EQ(1.0f, c);
        EXPECT_FLOAT_EQ(std::sin(static_cast<float>(pi * inp)), s);
        EXPECT_FLOAT_EQ(std::cos(static_cast<float>(pi * inp)), c);
    }
}

//---------------------------------------------------------------------------//

TEST(MathTest, signum)
{
    EXPECT_EQ(1, signum(2.0));
    EXPECT_EQ(-1, signum(-2.0));
    EXPECT_EQ(0, signum(0));
}

//---------------------------------------------------------------------------//

TEST(PortabilityTest, popcount)
{
    unsigned int x = 0xAA;
    EXPECT_EQ(4, popcount(x));

    x &= 0xF;
    EXPECT_EQ(2, popcount(x));
    x >>= 2;
    EXPECT_EQ(1, popcount(x));
    x >>= 2;
    EXPECT_EQ(0, popcount(x));
}

//---------------------------------------------------------------------------//

TEST(MathTest, TEST_IF_CELER_DEVICE(device))
{
    AlgorithmTestData data;
    data.num_threads = 32;

    auto mirror = [&] {
        // Build the input data
        HostVal<AlgorithmInputData> host_input;

        // Input for testing sincospi
        std::vector<double> pi_frac = {0.123, 0.0, 0.5, 1.0, 1.5};
        make_builder(&host_input.pi_frac)
            .insert_back(pi_frac.begin(), pi_frac.end());

        // Input for testing fastpow, etc.
        std::vector<double> a = {0.0, 0.0, 1234.0, 123.456, 1000.0, 2.0};
        std::vector<double> b = {1.0, 5.55042, 0.0, 1.0, 1.0 / 3.0, -5.0};
        make_builder(&host_input.a).insert_back(a.begin(), a.end());
        make_builder(&host_input.b).insert_back(b.begin(), b.end());

        // Copy to device
        return CollectionMirror<AlgorithmInputData>{std::move(host_input)};
    }();

    // Built the output data
    HostVal<AlgorithmOutputData> host_output;
    resize(&host_output, mirror.host_ref(), data.num_threads);

    // Copy to device
    AlgorithmOutputData<Ownership::value, MemSpace::device> device_output;
    device_output = host_output;

    // Launch kernel
    data.input = mirror.device_ref();
    data.output = device_output;
    alg_test(data);

    // Copy result back to host
    host_output = device_output;
    {
        // sincospi
        auto threads = range(ThreadId{host_output.sinpi.size()});
        static double const expected_sinpi[]
            = {0.37687101041216264, 0.0, 1.0, 0.0, -1.0};
        static double const expected_cospi[]
            = {0.92626575101906661, 1.0, 0.0, -1.0, 0.0};
        EXPECT_VEC_SOFT_EQ(expected_sinpi, host_output.sinpi[threads]);
        EXPECT_VEC_SOFT_EQ(expected_cospi, host_output.cospi[threads]);
    }
    {
        // fastpow
        auto threads = range(ThreadId{host_output.fastpow.size()});
        static double const expected_fastpow[]
            = {0, 0, 1, 123.456, 10, 0.03125};
        EXPECT_VEC_SOFT_EQ(expected_fastpow, host_output.fastpow[threads]);
    }
    {
        // hypot
        auto threads = range(ThreadId{host_output.hypot.size()});
        static double const expected_hypot[] = {
            1, 5.55042, 1234, 123.46004995949, 1000.0000555556, 5.3851648071345};
        EXPECT_VEC_SOFT_EQ(expected_hypot, host_output.hypot[threads]);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
