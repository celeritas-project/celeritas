//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/SpanUtils.test.cc
//---------------------------------------------------------------------------//
#include "corecel/math/SpanUtils.hh"

#include <cmath>

#include "corecel/cont/Span.hh"
#include "corecel/math/Algorithms.hh"
#include "celeritas/Constants.hh"

#include "TestMacros.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
inline constexpr auto m_pi = constants::pi;

using Dbl3 = Array<double, 3>;

enum
{
    X = 0,
    Y = 1,
    Z = 2
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST(SpanUtilsTest, copy)
{
    Dbl3 const src{1.0, 2.0, 3.0};
    Dbl3 dst{};
    copy(make_span(src), make_span(dst));
    EXPECT_VEC_EQ(src, dst);
}

TEST(SpanUtilsTest, fill)
{
    Dbl3 dst{};
    fill(7.0, make_span(dst));
    EXPECT_VEC_EQ((Dbl3{7, 7, 7}), dst);
}

TEST(SpanUtilsTest, axpy)
{
    Array<int, 3> const x{1, 3, 2};
    Array<int, 3> y{20, 30, 40};
    axpy(4, make_span(x), make_span(y));
    EXPECT_EQ(4 * 1 + 20, y[X]);
    EXPECT_EQ(4 * 3 + 30, y[Y]);
    EXPECT_EQ(4 * 2 + 40, y[Z]);
}

TEST(SpanUtilsTest, dot_product)
{
    Array<int, 2> const x{1, 3};
    Array<int, 2> const y{2, 4};
    EXPECT_EQ(1 * 2 + 3 * 4, dot_product(make_span(x), make_span(y)));
}

TEST(SpanUtilsTest, cross_product)
{
    Dbl3 const a{1, 0, 0};
    Dbl3 const b{0, 1, 0};
    Dbl3 result{};
    cross_product(make_span(a), make_span(b), make_span(result));
    EXPECT_VEC_SOFT_EQ((Dbl3{0, 0, 1}), result);
}

TEST(SpanUtilsTest, norm)
{
    Dbl3 const v{2, 3, 4};
    EXPECT_SOFT_EQ(std::sqrt(4.0 + 9.0 + 16.0), norm(make_span(v)));
}

TEST(SpanUtilsTest, make_unit_vector)
{
    Dbl3 const input{1, 2, 3};
    Dbl3 result{};
    make_unit_vector(make_span(input), make_span(result));
    double const factor = 1 / std::sqrt(1 + 4 + 9);
    EXPECT_VEC_SOFT_EQ((Dbl3{1 * factor, 2 * factor, 3 * factor}), result);
}

TEST(SpanUtilsTest, make_orthogonal)
{
    {
        Dbl3 const x{2, 0, 0};
        Dbl3 const y{0, 0, 1};
        Dbl3 result{};
        make_orthogonal(make_span(x), make_span(y), make_span(result));
        EXPECT_VEC_SOFT_EQ((Dbl3{2, 0, 0}), result);
    }
    {
        Dbl3 const x{2, 1, 3};
        Dbl3 const y{1, 0, 0};
        Dbl3 result{};
        make_orthogonal(make_span(x), make_span(y), make_span(result));
        EXPECT_VEC_SOFT_EQ((Dbl3{0, 1, 3}), result);
    }
}

TEST(SpanUtilsTest, is_soft_orthogonal)
{
    {
        Dbl3 const a{2, 0, 0};
        Dbl3 const b{0, 0, 1};
        EXPECT_TRUE(is_soft_orthogonal(make_span(a), make_span(b)));
    }
    {
        Dbl3 const a{2, 1, 3};
        Dbl3 const b{1, 0, 0};
        EXPECT_FALSE(is_soft_orthogonal(make_span(a), make_span(b)));
    }
}

TEST(SpanUtilsTest, is_soft_collinear)
{
    {
        Dbl3 const a{1, 0, 0};
        Dbl3 const b{1, 0, 0};
        EXPECT_TRUE(is_soft_collinear(make_span(a), make_span(b)));
    }
    {
        Dbl3 const a{1, 0, 0};
        Dbl3 const b{0, 1, 0};
        EXPECT_FALSE(is_soft_collinear(make_span(a), make_span(b)));
    }
}

TEST(SpanUtilsTest, distance)
{
    Dbl3 const x{3, 4, 5};
    Dbl3 const y{1, 1, 1};
    EXPECT_SOFT_EQ(std::sqrt(4.0 + 9.0 + 16.0),
                   distance(make_span(x), make_span(y)));
}

TEST(SpanUtilsTest, from_spherical)
{
    double constexpr costheta = 0.6;
    double constexpr phi = 0.8;
    double const sintheta = std::sqrt(1 - costheta * costheta);
    Dbl3 result{};
    from_spherical(costheta, phi, make_span(result));
    EXPECT_VEC_SOFT_EQ(
        (Dbl3{sintheta * std::cos(phi), sintheta * std::sin(phi), costheta}),
        result);
}

TEST(SpanUtilsTest, rotate)
{
    double const costheta = std::cos(2.0 / 3.0);
    double const phi = 2 * m_pi / 3.0;
    Dbl3 dir{};
    from_spherical(costheta, phi, make_span(dir));

    Dbl3 const rot = make_unit_vector(Dbl3{-1.1, 2.3, 0.9});
    Dbl3 result{};
    Dbl3 const& cdir = dir;
    rotate(make_span(cdir), make_span(rot), make_span(result));

    // Result must be a unit vector
    EXPECT_SOFT_EQ(1.0, dot_product(result, result));

    // Must match the Array-based rotate
    EXPECT_VEC_SOFT_EQ(::celeritas::rotate(dir, rot), result);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
