//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file TestMacros.test.cc
//---------------------------------------------------------------------------//
#include "TestMacros.hh"

#include <limits>

#include "corecel/cont/Array.hh"
#include "corecel/io/ScopedStreamRedirect.hh"

#include "celeritas_test.hh"

using Dbl2 = celeritas::Array<double, 2>;
using VecInt = std::vector<int>;
using VecDbl = std::vector<double>;
using VecFlt = std::vector<float>;

constexpr double inf = std::numeric_limits<double>::infinity();

namespace celeritas
{
namespace test
{

//---------------------------------------------------------------------------//

TEST(PrintExpected, example)
{
    std::string out_result = [] {
        ScopedStreamRedirect redirect_out(&std::cout);
        std::cout << '\n';

        std::vector<int> values = {1, 2, 3};
        PRINT_EXPECTED(values);

        using Limits_t = std::numeric_limits<double>;
        double const more[]
            = {.5, .001, Limits_t::infinity(), Limits_t::quiet_NaN()};
        PRINT_EXPECTED(more);

        char const* const cstrings[] = {"one", "three", "five"};
        PRINT_EXPECTED(cstrings);

        std::string const strings[] = {"a", "", "special\nchars\t"};
        PRINT_EXPECTED(strings);

        return redirect_out.str();
    }();
    out_result.push_back('\n');
    EXPECT_EQ(R"(
static int const expected_values[] = {1, 2, 3};
static double const expected_more[] = {0.5, 0.001, inf, nan};
static char const* const expected_cstrings[] = {"one", "three", "five"};
static std::string const expected_strings[] = {"a", "", "special\nchars\t"};
)",
              out_result);
}
}  // namespace test

namespace testdetail
{
//---------------------------------------------------------------------------//
TEST(IsSoftEquiv, successes)
{
    EXPECT_TRUE(
        IsSoftEquiv("1.", "1. + 1.e-13", "1.e-12", 1., 1. + 1.e-13, 1.e-12));
    EXPECT_TRUE(IsSoftEquiv("0.", "1.e-15", "1.e-12", 0., 1.e-15, 1.e-12));
    EXPECT_TRUE(
        IsSoftEquiv("1.e-15", "-1.2e-15", "1.e-12", 1.e-15, -1.2e-15, 1.e-12));
    EXPECT_TRUE(IsSoftEquiv("10.", "10.05", "1.e-2", 10., 10.05, 1.e-2));

    // no default tolerance
    EXPECT_TRUE(IsSoftEquiv("1.", "1. + 1.e-13", 1., 1. + 1.e-13));

    // Check infinities
    EXPECT_TRUE(IsSoftEquiv("inf", "inf", inf, inf));
    EXPECT_TRUE(IsSoftEquiv("-inf", "-inf", -inf, -inf));
}

TEST(IsSoftEquiv, failures)
{
    EXPECT_FALSE(
        IsSoftEquiv("1.", "1. + 1.e-13", "1.e-14", 1., 1. + 1.e-13, 1.e-14));
    // relative threshold not met from zero
    EXPECT_FALSE(IsSoftEquiv("0.", "2.e-13", "1.e-13", 0., 2.e-13, 1.e-13));
    // relative not met
    EXPECT_FALSE(IsSoftEquiv("10.", "10.05", "1.e-3", 10., 10.05, 1.e-3));

    // no default tolerance
    EXPECT_FALSE(IsSoftEquiv("1.", "1. + 1.e-11", 1., 1. + 1.e-11));

    // Mismatched infinities
    EXPECT_FALSE(IsSoftEquiv("inf", "-inf", inf, -inf));
}

//---------------------------------------------------------------------------//
TEST(IsVecSoftEquiv, successes_double)
{
    VecDbl expected, actual;
    double expected_array[] = {1.005, 0.009};

    // empty vectors
    EXPECT_TRUE(
        IsVecSoftEquiv("expected", "actual", "0.01", expected, actual, 0.01));

    // keep the same size
    expected.push_back(1.);
    actual.push_back(1.005);
    EXPECT_TRUE(
        IsVecSoftEquiv("expected", "actual", "0.01", expected, actual, 0.01));

    // test comparison against zero
    expected.push_back(0.);
    actual.push_back(0.009);
    EXPECT_FALSE(
        IsVecSoftEquiv("expected", "actual", "0.01", expected, actual, 0.01));
    EXPECT_TRUE(IsVecSoftEquiv(
        "expected", "actual", "soft", expected, actual, (SoftEqual{0.01, 0.01})));

    EXPECT_TRUE(IsVecSoftEquiv(
        "expected_array", "actual", "0.01", expected_array, actual, 0.01));

    // no default tolerance
    expected.clear();
    expected.push_back(8.);
    actual.clear();
    actual.push_back(8. * (1 + 1.e-13));
    EXPECT_VEC_SOFT_EQ(expected, actual);

    // infinity
    expected = {inf, -inf};
    actual = expected;
    EXPECT_VEC_SOFT_EQ(expected, actual);
}

TEST(IsVecSoftEquiv, floats)
{
    // Compare identical values, array vs vector
    float const expected_array[] = {1.f, 3.f, 5.f};
    VecFlt actual(expected_array, expected_array + 3);
    EXPECT_VEC_SOFT_EQ(expected_array, actual);

    // Compare vector vs array
    EXPECT_VEC_SOFT_EQ(actual, expected_array);
    EXPECT_VEC_NEAR(actual, expected_array, 1.e-3f);
    EXPECT_VEC_NEAR(actual, expected_array, (SoftEqual{1.e-3f, 1e-5f}));

    // Compare floating point tolerance is not 1e-12
    float actual_array[] = {1.000001f, 3.000001f, 5.000001f};

    EXPECT_TRUE(IsVecSoftEquiv(
        "expected_array", "actual", expected_array, actual_array));

    // The floating point type should convert from double to float
    EXPECT_TRUE(IsVecSoftEquiv(
        "expected_array", "actual", "1e-6", expected_array, actual_array, 1e-6f));

    actual_array[0] = 1.0001f;
    EXPECT_FALSE(IsVecSoftEquiv(
        "expected_array", "actual", expected_array, actual_array));
}

TEST(IsVecSoftEquiv, mixed)
{
    // Single-precision is expected
    VecFlt expected_flt = {0.125f, 1.123f, 5.f};
    VecDbl expected_dbl = {0.125, 1.123, 5.};

    // Convert to double
    VecDbl upcast(expected_flt.begin(), expected_flt.end());

    EXPECT_FALSE(
        IsVecSoftEquiv("expected_dbl", "upcast", expected_dbl, upcast));

    EXPECT_TRUE(IsVecSoftEquiv(
        "expected_flt", "expected_dbl", expected_flt, expected_dbl));

    // Comparisons are cast to floating point first
    EXPECT_TRUE(IsVecSoftEquiv("expected_flt",
                               "expected_dbl",
                               "1.e-12",
                               expected_flt,
                               expected_dbl,
                               1.e-12f));

    VecFlt actual_flt = {0.126f, 1.123f, 5.f};

    EXPECT_FALSE(IsVecSoftEquiv(
        "expected_flt", "actual_flt", expected_flt, actual_flt));

    EXPECT_TRUE(IsVecSoftEquiv(
        "expected_flt", "actual_flt", ".1", expected_flt, actual_flt, 0.1f));
}

TEST(IsVecSoftEquiv, nested)
{
    std::vector<Dbl2> const expected[] = {
        {{0.12, 0.23}, {0.34, 0.45}},
        {{0.56, 0.67}, {0.78, 0.89}},
        {{0.91, 0.12}, {0.23, 0.34}},
    };
    std::vector<std::vector<Dbl2>> actual = {
        {{0.12, 0.23}, {0.34, 0.45}},
        {{0.56, 0.671}},
        {{0.91, 0.12}, {0.23, 0.34}},
    };

    // Different sizes
    EXPECT_FALSE(
        IsVecSoftEquiv("expected", "actual", "0.01", expected, actual, 0.01));
    EXPECT_TRUE(IsVecSoftEquiv(
        "expected[0]", "actual[0]", "0.01", expected[0], actual[0], 0.01));

    // Test within tolerance
    actual[1].push_back({0.78, 0.89});
    EXPECT_TRUE(
        IsVecSoftEquiv("expected", "actual", "0.01", expected, actual, 0.01));
    EXPECT_FALSE(IsVecSoftEquiv(
        "expected", "actual", "0.001", expected, actual, 0.001));
    EXPECT_TRUE(IsVecSoftEquiv(
        "expected", "actual", "0.01", expected, actual, (SoftEqual{0.01, 0.01})));

    // Identical
    actual[1][0][1] = 0.67;
    EXPECT_VEC_SOFT_EQ(expected, actual);
}

// NOTE: these should fail to compile and produce relatively understandable
// error messages.
#if 0
//---------------------------------------------------------------------------//
TEST(IsSoftEquiv, ints)
{
    EXPECT_SOFT_EQ(100, 101);
    EXPECT_SOFTEQ(100, 101, 0.05);
}

TEST(IsVecSoftEquiv, ints)
{
    const int a[] = {1,2,3,4};
    const unsigned int b[] = {2,2,3,4};

    EXPECT_VEC_SOFT_EQ(b, a);
}
#endif

//---------------------------------------------------------------------------//
// Note: to test what the output looks like, just change EXPECT_FALSE to
// EXPECT_TRUE, or do cout << IsVecSoftEquiv(...).message() << endl;
TEST(IsVecSoftEquiv, failures)
{
    VecDbl expected, actual;

    // wrong sized vectors
    expected.push_back(1.);
    EXPECT_FALSE(
        IsVecSoftEquiv("expected", "actual", "0.01", expected, actual, 0.01));

    // single wrong value
    actual.push_back(0.);
    EXPECT_FALSE(
        IsVecSoftEquiv("expected", "actual", "0.01", expected, actual, 0.01));

    // truncating long expression values
    EXPECT_FALSE(IsVecSoftEquiv(
        "expectedddddddddd", "actualllllllllll", "0.01", expected, actual, 0.01));

    // multiple wrong values
    expected.push_back(100.);
    actual.push_back(102.);
    EXPECT_FALSE(
        IsVecSoftEquiv("expected", "actual", "0.01", expected, actual, 0.01));

    // Ten wrong values
    expected.assign(10, 5.);
    actual.assign(10, 6.);
    EXPECT_FALSE(
        IsVecSoftEquiv("expected", "actual", "0.01", expected, actual, 0.01));

    // Uncomment this line to test output
    // EXPECT_VEC_NEAR(expected, actual, 0.01);

    // 100 wrong values (should truncate)
    expected.assign(100, 5.);
    actual.assign(100, 6.);
    EXPECT_FALSE(
        IsVecSoftEquiv("expected", "actual", "0.01", expected, actual, 0.01));

    // A couple of wrong values in a large array
    expected.assign(200, 10.);
    actual.assign(200, 10.);
    actual[0] = 11.;
    actual[5] = 1. + 1.e-11;
    actual[100] = 3.;
    actual[150] = 2.;
    EXPECT_FALSE(IsVecSoftEquiv(
        "expected", "actual", "1.e-12", expected, actual, 1.e-12));

    // infinity
    expected = {inf, -inf, inf, inf};
    actual = {-inf, inf, 0, 1};
    // EXPECT_VEC_SOFT_EQ(expected, actual);
    EXPECT_FALSE(IsVecSoftEquiv(
        "expected", "actual", "1.e-12", expected, actual, 1.e-12));
}

//---------------------------------------------------------------------------//
TEST(IsVecEq, successes)
{
    VecInt expected, actual;

    // empty vectors
    EXPECT_TRUE(IsVecEq("expected", "actual", expected, actual));

    // keep the same size
    expected.push_back(1);
    actual.push_back(1);
    EXPECT_TRUE(IsVecEq("expected", "actual", expected, actual));

    // test comparison against zero
    expected.push_back(0);
    actual.push_back(0);
    EXPECT_TRUE(IsVecEq("expected", "actual", expected, actual));

    expected.clear();
    expected.push_back(8);
    actual.clear();
    actual.push_back(8);
    EXPECT_VEC_EQ(expected, actual);

    int static_array[] = {8, 0, 3};
    actual.clear();
    actual.push_back(8);
    actual.push_back(0);
    actual.push_back(3);

    EXPECT_VEC_EQ(static_array, actual);

    // Test nested containers
    std::vector<VecInt> expected_vec, actual_vec;
    EXPECT_VEC_EQ(expected_vec, actual_vec);

    expected_vec = {{{1}, {2, 3}, {4, 5, 5}}};
    actual_vec = {{{1}, {2, 3}, {4, 5, 5}}};
    EXPECT_TRUE(
        IsVecEq("expected_vec", "actual_vec", expected_vec, actual_vec));
    EXPECT_VEC_EQ(expected_vec, actual_vec);
}

// Note: to test what the output looks like, just change EXPECT_FALSE to
// EXPECT_TRUE, or do cout << IsVecEq(...).message() << endl;
TEST(IsVecEq, failures)
{
    VecInt expected, actual;

    int static_array[] = {1, 100};

    // wrong sized vectors
    expected.push_back(1);
    EXPECT_FALSE(IsVecEq("expected", "actual", expected, actual));
    EXPECT_FALSE(IsVecEq("static_array", "actual", static_array, actual));

    // single wrong value
    actual.push_back(0);
    EXPECT_FALSE(IsVecEq("expected", "actual", expected, actual));

    // truncating long expression values
    EXPECT_FALSE(
        IsVecEq("expectedddddddddd", "actualllllllllll", expected, actual));

    // multiple wrong values
    expected.push_back(100);
    actual.push_back(102);
    EXPECT_FALSE(IsVecEq("expected", "actual", expected, actual));
    EXPECT_FALSE(IsVecEq("static_array", "actual", static_array, actual));

    // Ten wrong values
    expected.assign(10, 5);
    actual.assign(10, 6);
    EXPECT_FALSE(IsVecEq("expected", "actual", expected, actual));

    // Uncomment this line to test output
    // EXPECT_VEC_EQ(expected, actual);

    // 100 wrong values (should truncate)
    expected.assign(100, 5);
    actual.assign(100, 6);
    EXPECT_FALSE(IsVecEq("expected", "actual", expected, actual));

    // A couple of wrong values in a large array
    expected.assign(200, 10);
    actual.assign(200, 10);
    actual[0] = 1;
    actual[5] = 1;
    actual[100] = 3;
    actual[150] = 2;
    EXPECT_FALSE(IsVecEq("expected", "actual", expected, actual));

    // Test nested containers
    std::vector<VecInt> expected_vec = {{0, 1}, {2, 3, 4}};
    std::vector<VecInt> actual_vec = {{0, 1}};
    EXPECT_FALSE(
        IsVecEq("expected_vec", "actual_vec", expected_vec, actual_vec));

    actual_vec.push_back({2, 3});
    EXPECT_FALSE(
        IsVecEq("expected_vec", "actual_vec", expected_vec, actual_vec));

    actual_vec[1].push_back(5);
    EXPECT_FALSE(
        IsVecEq("expected_vec", "actual_vec", expected_vec, actual_vec));
}

struct Foo
{
    int val{0};
};

struct FooTol
{
    int val{0};
};

inline ::testing::AssertionResult
IsRefEq(char const* expr1, char const* expr2, Foo const& val1, Foo const& val2)
{
    ::celeritas::test::AssertionHelper result(expr1, expr2);

    if (val1.val != val2.val)
    {
        result.fail() << "  foo: " << val1.val << " != " << val2.val;
    }
    return result;
}

inline ::testing::AssertionResult IsRefEq(char const* expr1,
                                          char const* expr2,
                                          char const* tolexpr,
                                          Foo const& val1,
                                          Foo const& val2,
                                          FooTol const& tol)
{
    ::celeritas::test::AssertionHelper result(expr1, expr2);

    if (std::abs(val1.val - val2.val) >= tol.val)
    {
        result.fail() << "  foo: " << val1.val << " != " << val2.val
                      << " within " << tolexpr << " (" << tol.val << ")";
    }
    return result;
}

TEST(IsRefEq, successes)
{
    Foo ref;
    ref.val = 2;

    EXPECT_REF_EQ(ref, ref);
    EXPECT_REF_NEAR(ref, ref, FooTol{123});

    // Vectors
    static Foo const vexpected[] = {Foo{2}, Foo{2}};
    std::vector<Foo> vactual = {ref, ref};
    EXPECT_REF_EQ(vexpected, vactual);
    EXPECT_REF_NEAR(vexpected, vactual, FooTol{123});
}

TEST(IsRefEq, failures)
{
    Foo ref;
    ref.val = 2;

    EXPECT_FALSE(IsRefEq("expected", "actual", ref, Foo{1}));
    EXPECT_FALSE(IsRefEq("expected", "actual", "tol", ref, Foo{20}, FooTol{5}));

    // Vectors of different size
    {
        static Foo const vexpected[] = {Foo{2}, Foo{2}};
        std::vector<Foo> vactual = {ref, ref, ref};
        EXPECT_FALSE(IsRefEq("expected", "actual", vexpected, vactual));
        EXPECT_FALSE(IsRefEq(
            "expected", "actual", "tol", vexpected, vactual, FooTol{5}));
    }

    // Vectors of same size
    {
        std::vector<Foo> vexpected(100, ref);
        std::vector<Foo> vactual(100, Foo{50});
        EXPECT_FALSE(IsRefEq("expected", "actual", vexpected, vactual));
        EXPECT_FALSE(IsRefEq(
            "expected", "actual", "tol", vexpected, vactual, FooTol{20}));
    }
}

//---------------------------------------------------------------------------//
}  // namespace testdetail
}  // namespace celeritas
