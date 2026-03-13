//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/ArrayUtils.test.cc
//---------------------------------------------------------------------------//
#include "corecel/math/ArrayUtils.hh"

#include "corecel/io/StreamToString.hh"
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

using Real3 = Array<real_type, 3>;
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

TEST(ArrayUtilsTest, io)
{
    Array<int, 3> x{1, 3, 2};
    EXPECT_EQ("{1,3,2}", stream_to_string(x));
}

TEST(ArrayUtilsTest, axpy)
{
    Array<int, 3> x{1, 3, 2};
    Array<int, 3> y{20, 30, 40};

    axpy(4, x, &y);
    EXPECT_EQ(4 * 1 + 20, y[X]);
    EXPECT_EQ(4 * 3 + 30, y[Y]);
    EXPECT_EQ(4 * 2 + 40, y[Z]);
}

TEST(ArrayUtilsTest, dot_product)
{
    Array<int, 2> x{1, 3};
    Array<int, 2> y{2, 4};

    EXPECT_EQ(1 * 2 + 3 * 4, dot_product(x, y));
}

TEST(ArrayUtilsTest, norm)
{
    EXPECT_SOFT_EQ(std::sqrt(4.0 + 9.0 + 16.0),
                   norm(Array<double, 3>{2, 3, 4}));
}

TEST(ArrayUtilsTest, distance)
{
    EXPECT_SOFT_EQ(
        std::sqrt(4.0 + 9.0 + 16.0),
        distance(Array<double, 3>{3, 4, 5}, Array<double, 3>{1, 1, 1}));
}

TEST(ArrayUtilsTest, make_unit_vector)
{
    Dbl3 direction = make_unit_vector(Dbl3{1, 2, 3});
    double const norm = 1 / std::sqrt(1 + 4 + 9);
    EXPECT_VEC_SOFT_EQ((Dbl3{1 * norm, 2 * norm, 3 * norm}), direction);

    EXPECT_TRUE(std::isnan(make_unit_vector(Dbl3{0, 0, 0})[0]));
}

TEST(ArrayUtilsTest, make_orthogonal)
{
    EXPECT_VEC_SOFT_EQ((Dbl3{2, 0, 0}),
                       make_orthogonal(Dbl3{2, 0, 0}, Dbl3{0, 0, 1}));
    EXPECT_VEC_SOFT_EQ((Dbl3{0, 1, 3}),
                       make_orthogonal(Dbl3{2, 1, 3}, Dbl3{1, 0, 0}));
    EXPECT_VEC_SOFT_EQ(
        (Dbl3{0.5, -0.5, 3}),
        make_orthogonal(Dbl3{2, 1, 3}, make_unit_vector(Dbl3{1, 1, 0})));
}

TEST(ArrayUtilsTest, is_soft_orthogonal)
{
    EXPECT_TRUE(is_soft_orthogonal(Dbl3{2, 0, 0}, Dbl3{0, 0, 1}));
    EXPECT_TRUE(is_soft_orthogonal(Dbl3{0, 1, 0}, Dbl3{1, 0, 0}));
    EXPECT_TRUE(is_soft_orthogonal(Dbl3{0, 1, 1}, Dbl3{1, 0, 0}));
    EXPECT_TRUE(is_soft_orthogonal(Dbl3{1e-13, 1, 0}, Dbl3{1, 0, 0}));

    EXPECT_FALSE(is_soft_orthogonal(Dbl3{2, 1, 3}, Dbl3{1, 0, 0}));
    EXPECT_FALSE(is_soft_orthogonal(Dbl3{1e-6, 1, 0}, Dbl3{1, 0, 0}));
}

TEST(ArrayUtilsTest, is_soft_collinear)
{
    constexpr double x2{1 / constants::sqrt_two};
    constexpr double x3{1 / constants::sqrt_three};

    EXPECT_TRUE(is_soft_collinear(Dbl3{1, 0, 0}, Dbl3{1, 0, 0}));
    EXPECT_TRUE(is_soft_collinear(Dbl3{0, x2, x2}, Dbl3{0, x2, x2}));
    EXPECT_TRUE(is_soft_collinear(Dbl3{-x3, x3, -x3}, Dbl3{-x3, x3, -x3}));
    EXPECT_TRUE(is_soft_collinear(make_unit_vector(Dbl3{x3 + 1e-12, x3, x3}),
                                  Dbl3{x3, x3, x3}));
    EXPECT_TRUE(is_soft_collinear(make_unit_vector(Dbl3{x3 - 1e-12, x3, x3}),
                                  Dbl3{x3, x3, x3}));

    EXPECT_FALSE(is_soft_collinear(Dbl3{1, 0, 0}, Dbl3{0, 1, 0}));
    EXPECT_FALSE(is_soft_collinear(Dbl3{x3, x3, x3}, Dbl3{x3, x3, -x3}));
    EXPECT_FALSE(is_soft_collinear(make_unit_vector(Dbl3{x3 + 1e-4, x3, x3}),
                                   Dbl3{x3, x3, x3}));
    EXPECT_FALSE(is_soft_collinear(make_unit_vector(Dbl3{x3 - 1e-4, x3, x3}),
                                   Dbl3{x3, x3, x3}));

    if (CELERITAS_DEBUG)
    {
        EXPECT_THROW(is_soft_collinear(Dbl3{1, 0, 0}, Dbl3{1, 1, 0}),
                     DebugError);
        EXPECT_THROW(is_soft_collinear(Dbl3{x3, x3, 0}, Dbl3{1, 0, 0}),
                     DebugError);
    }
}

TEST(ArrayUtilsTest, is_soft_unit_vector)
{
    Real3 dir = make_unit_vector(Real3{1, 2, 3});
    EXPECT_SOFT_EQ(1, dot_product(dir, dir));
    EXPECT_TRUE(is_soft_unit_vector(dir));
    constexpr real_type eps = SoftEqual<real_type>{}.rel();
    dir[0] += eps;
    EXPECT_TRUE(is_soft_unit_vector(dir));
    dir[1] += eps;
    EXPECT_TRUE(is_soft_unit_vector(dir));
    dir[2] -= eps;
    EXPECT_TRUE(is_soft_unit_vector(dir));
    dir[0] += 10 * eps;
    EXPECT_FALSE(is_soft_unit_vector(dir));

    // Test nan
    constexpr auto nan = std::numeric_limits<real_type>::quiet_NaN();
    EXPECT_FALSE(is_soft_unit_vector(Real3{nan, 1, 0}));
}

TEST(ArrayUtilsTest, rotate)
{
    Dbl3 vec = make_unit_vector(Dbl3{-1.1, 2.3, 0.9});

    // transform through some directions
    double costheta = std::cos(2.0 / 3.0);
    double sintheta = std::sqrt(1.0 - ipow<2>(costheta));
    double phi = 2 * m_pi / 3.0;

    double a = 1.0 / sqrt(1.0 - vec[Z] * vec[Z]);
    Dbl3 expected
        = {vec[X] * costheta + vec[Z] * vec[X] * sintheta * cos(phi) * a
               - vec[Y] * sintheta * sin(phi) * a,
           vec[Y] * costheta + vec[Z] * vec[Y] * sintheta * cos(phi) * a
               + vec[X] * sintheta * sin(phi) * a,
           vec[Z] * costheta - sintheta * cos(phi) / a};

    auto scatter = from_spherical(costheta, phi);
    EXPECT_VEC_SOFT_EQ(expected, rotate(scatter, vec));

    // Transform degenerate vector along y
    expected = {-sintheta * cos(phi), sintheta * sin(phi), -costheta};
    EXPECT_VEC_SOFT_EQ(expected, rotate(scatter, {0.0, 0.0, -1.0}));

    expected = {sintheta * cos(phi), sintheta * sin(phi), costheta};
    EXPECT_VEC_SOFT_EQ(expected, rotate(scatter, {0.0, 0.0, 1.0}));

    // Transform almost degenerate vector
    vec = make_unit_vector(Dbl3{3e-8, 4e-8, 1});
    EXPECT_VEC_SOFT_EQ(
        (Dbl3{-0.613930085414816, 0.0739664834328671, 0.785887275346237}),
        rotate(scatter, vec));

    // Transform a range of more almost-degenerate vectors
    for (auto eps : {1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8})
    {
        vec = make_unit_vector(Dbl3{-eps, 2 * eps, 1 - eps * eps});
        Dbl3 result = rotate(scatter, vec);
        EXPECT_SOFT_EQ(1.0, dot_product(result, result))
            << "for eps=" << eps << " => vec=" << vec;
    }

    // Test non-unit result failure from KN demo kernel
    scatter = {-0.49359585864063, 0.814702199615169, -0.304341016419122};
    EXPECT_SOFT_EQ(0.0, dot_product(scatter, scatter) - 1);
    vec = {0.00208591611691868, 0.00328080730344347, 0.999992442600138};
    EXPECT_SOFT_EQ(0.0, dot_product(vec, vec) - 1);
    {
        auto result = rotate(scatter, vec);
        EXPECT_SOFT_EQ(0.0, dot_product(result, result) - 1);
        // The expected value below is from using the regular/non-fallback
        // calculation normalized to a unit vector
        EXPECT_VEC_NEAR(
            (Dbl3{-0.952973648767149, 0.0195839636531213, -0.302419730049247}),
            result,
            20 * SoftEqual<double>{}.rel());
    }

    // Switch scattered z direction
    costheta *= -1;
    scatter = from_spherical(costheta, phi);

    expected = {-sintheta * cos(phi), sintheta * sin(phi), -costheta};
    EXPECT_VEC_SOFT_EQ(expected, rotate(scatter, {0.0, 0.0, -1.0}));

    expected = {sintheta * cos(phi), sintheta * sin(phi), costheta};
    vec = rotate(scatter, {0.0, 0.0, 1.0});
    EXPECT_VEC_SOFT_EQ(expected, vec);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
