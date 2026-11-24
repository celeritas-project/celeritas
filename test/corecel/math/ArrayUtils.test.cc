//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/ArrayUtils.test.cc
//---------------------------------------------------------------------------//
#include "corecel/math/ArrayUtils.hh"

#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

#include "corecel/Constants.hh"
#include "corecel/io/StreamUtils.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/ArrayOperators.hh"
#include "corecel/math/SoftEqual.hh"

#include "TestMacros.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
using Real3 = Array<real_type, 3>;
using Dbl3 = Array<double, 3>;

inline constexpr int X = 0;
inline constexpr int Y = 1;
inline constexpr int Z = 2;

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

TEST(RotateTest, basic)
{
    Dbl3 vec = make_unit_vector(Dbl3{-1.1, 2.3, 0.9});

    // Transform through some directions
    double costheta = std::cos(2.0 / 3.0);
    double sintheta = std::sqrt(1.0 - ipow<2>(costheta));
    double phi = 2 * constants::pi / 3.0;

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

    {
        // Test non-unit result failure from KN demo kernel
        scatter = {-0.49359585864063, 0.814702199615169, -0.304341016419122};
        EXPECT_SOFT_EQ(0.0, dot_product(scatter, scatter) - 1);
        vec = {0.00208591611691868, 0.00328080730344347, 0.999992442600138};
        EXPECT_SOFT_EQ(0.0, dot_product(vec, vec) - 1);

        auto result = rotate(scatter, vec);
        EXPECT_SOFT_EQ(0.0, dot_product(result, result) - 1);
        // The expected value below is from using the regular/non-fallback
        // calculation normalized to a unit vector
        EXPECT_VEC_NEAR(
            (Dbl3{-0.952973648767149, 0.0195839636531213, -0.302419730049247}),
            result,
            20 * SoftEqual<double>{}.rel());
    }
}

TEST(RotateTest, degenerate)
{
    // Transform a range of more almost-degenerate vectors
    auto scatter = from_spherical(std::cos(2.0 / 3.0), 2.0 * constants::pi / 3);
    static_assert(std::is_same_v<decltype(scatter), Dbl3>);
    for (auto eps : {1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8})
    {
        Dbl3 vec = make_unit_vector(Dbl3{-eps, 2 * eps, 1 - eps * eps});
        Dbl3 result = rotate(scatter, vec);
        EXPECT_SOFT_EQ(1.0, dot_product(result, result))
            << "for eps=" << eps << " => vec=" << vec;
    }

    // Transform a different almost degenerate vector
    EXPECT_VEC_SOFT_EQ(
        (Dbl3{-0.613930085414816, 0.0739664834328671, 0.785887275346237}),
        rotate(scatter, make_unit_vector(Dbl3{3e-8, 4e-8, 1})));

    // Normalization not quite right
    // constexpr auto eps = std::numeric_limits<double>::epsilon();
    constexpr auto eps = SoftEqual<double>{}.rel();
    scatter = {-0.25938973121662295, 0.67784930294796419, 0.68792244463480556};
    EXPECT_VEC_SOFT_EQ(
        (Dbl3{-0.25938973121662295, 0.67784930294796419, 0.68792244463480556}),
        rotate(scatter, Dbl3{0, 0, 1 - eps}));
    EXPECT_VEC_SOFT_EQ(
        (Dbl3{0.259388758357873, -0.677849302947964, 0.687922811462536}),
        rotate(scatter, Dbl3{-eps, 0, 1 - eps}));
    EXPECT_VEC_SOFT_EQ(
        (Dbl3{-0.677849302947964, -0.259388758357873, 0.687922811462536}),
        rotate(scatter, Dbl3{0, eps, 1 - eps}));
    EXPECT_VEC_SOFT_EQ(scatter, rotate(scatter, Dbl3{0, 0, 1 + eps}));

    // NOTE: this array direction is "soft equal" to 1 but currently fails
    // because of sqrt(-epsilon) getting propagated into final answer
    auto nan_result = rotate(scatter, Dbl3{eps, 0, 1 + eps});
    EXPECT_TRUE(std::isnan(nan_result[0]) && std::isnan(nan_result[1])
                && std::isnan(nan_result[2]))
        << nan_result;

    // Transform fully degenerate vector
    Dbl3 const dir{-0.613930085414816, 0.0739664834328671, 0.785887275346237};
    EXPECT_VEC_SOFT_EQ(dir, rotate(dir, Dbl3{0, 0, 1}));
    EXPECT_VEC_SOFT_EQ(-dir, rotate(-dir, Dbl3{0, 0, 1}));
    EXPECT_VEC_SOFT_EQ((Dbl3{-dir[0], dir[1], -dir[2]}),
                       rotate(dir, Dbl3{0, 0, -1}));
}

TEST(RotateTest, neg_z)
{
    // Transform from negative Z direction
    double costheta = -std::cos(2.0 / 3.0);
    double sintheta = std::sqrt(1.0 - ipow<2>(costheta));
    double phi = 2 * constants::pi / 3.0;

    Dbl3 scatter = from_spherical(costheta, phi);

    Dbl3 expected = {-sintheta * cos(phi), sintheta * sin(phi), -costheta};
    EXPECT_VEC_SOFT_EQ(expected, rotate(scatter, {0.0, 0.0, -1.0}));

    expected = {sintheta * cos(phi), sintheta * sin(phi), costheta};
    EXPECT_VEC_SOFT_EQ(expected, rotate(scatter, {0.0, 0.0, 1.0}));
}

TEST(RotateTest, near_z)
{
    // Incident directions close enough to +/-z to use the normalized x/y
    // calculation, in all four x/y quadrants
    for (Dbl3 const& v : {Dbl3{3e-3, 4e-3, 1},
                          Dbl3{-3e-3, 4e-3, 1},
                          Dbl3{-3e-3, -4e-3, 1},
                          Dbl3{3e-3, -4e-3, 1},
                          Dbl3{3e-3, -4e-3, -1},
                          Dbl3{-3e-3, -4e-3, -1}})
    {
        Dbl3 const rot = make_unit_vector(v);
        SCOPED_TRACE(rot);

        // Unrotated direction maps onto the rotation axis
        EXPECT_VEC_SOFT_EQ(rot, rotate(Dbl3{0, 0, 1}, rot));

        // Polar angle with respect to the rotation axis is preserved
        for (double costheta : {0.9, 0.0, -0.5})
        {
            for (double phi : {0.0, 1.0, 4.0})
            {
                auto result = rotate(from_spherical(costheta, phi), rot);
                EXPECT_SOFT_EQ(costheta, dot_product(result, rot));
            }
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Rotate with an independently constructed, branch-free rotation matrix.
 *
 * The columns are the images of the x, y, and z unit vectors: \f$ (\cos\theta
 * \cos\phi, \cos\theta \sin\phi, -\sin\theta) \f$, \f$ (-\sin\phi, \cos\phi,
 * 0) \f$, and \c rot itself. The angles come from \c atan2 so the result is
 * accurate arbitrarily close to the z axis.
 */
Dbl3 reference_rotate(Dbl3 const& dir, Dbl3 const& rot)
{
    double const theta = std::atan2(std::hypot(rot[X], rot[Y]), rot[Z]);
    double const phi = std::atan2(rot[Y], rot[X]);
    Dbl3 const ex{std::cos(theta) * std::cos(phi),
                  std::cos(theta) * std::sin(phi),
                  -std::sin(theta)};
    Dbl3 const ey{-std::sin(phi), std::cos(phi), 0.0};

    Dbl3 result;
    for (int i = 0; i != 3; ++i)
    {
        result[i] = dir[X] * ex[i] + dir[Y] * ey[i] + dir[Z] * rot[i];
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Check that rotation about \c rot is a proper rotation that maps z to \c rot.
 *
 * These checks target the failure modes of past bugs in \c rotate:
 * - z must map onto \c rot (the incident direction, and hence momentum, is
 *   preserved);
 * - the images of x and y must complete a right-handed frame (a proper
 *   rotation, not a reflection);
 * - the polar angle about \c rot must equal the scattering angle, including
 *   its sign;
 * - the result must match a branch-free reference, including its length and
 *   azimuth.
 */
template<class T>
void check_rotate(Array<T, 3> const& rot, double tol)
{
    using ArrT = Array<T, 3>;
    auto rotate_dbl = [&rot](ArrT const& dir) {
        return static_array_cast<double>(rotate(dir, rot));
    };
    Dbl3 const drot = static_array_cast<double>(rot);

    Dbl3 const ex = rotate_dbl(ArrT{1, 0, 0});
    Dbl3 const ey = rotate_dbl(ArrT{0, 1, 0});
    Dbl3 const ez = rotate_dbl(ArrT{0, 0, 1});
    EXPECT_LE(distance(drot, ez), tol) << ez;
    EXPECT_LE(distance(ez, cross_product(ex, ey)), tol) << ex << ey;

    for (double costheta : {1.0, 0.9, 0.0, -0.5, -1.0})
    {
        for (double phi : {0.0, 1.0, 4.0})
        {
            ArrT const dir
                = static_array_cast<T>(from_spherical(costheta, phi));
            Dbl3 const actual = rotate_dbl(dir);
            EXPECT_NEAR(costheta, dot_product(actual, drot), tol);
            EXPECT_LE(distance(reference_rotate(static_array_cast<double>(dir),
                                                drot),
                               actual),
                      tol)
                << "costheta=" << costheta << ", phi=" << phi;
        }
    }
}

//---------------------------------------------------------------------------//
//! Rotation axes at a given distance from +/-z in all x/y quadrants
std::vector<Dbl3> near_z_axes(double sintheta)
{
    double const costheta = std::sqrt(1 - ipow<2>(sintheta));
    std::vector<Dbl3> result;
    for (double zsign : {1.0, -1.0})
    {
        for (double ysign : {1.0, -1.0})
        {
            for (double xsign : {1.0, -1.0})
            {
                result.push_back({xsign * 0.6 * sintheta,
                                  ysign * 0.8 * sintheta,
                                  zsign * costheta});
            }
        }
    }
    return result;
}

template<class T>
class RotateTypedTest : public ::testing::Test
{
  protected:
    using ArrT = Array<T, 3>;

    //! Tolerance: just above the threshold, sqrt(1 - z^2) loses about
    //! eps / sin^2(theta) relative precision (measured ~1e-5 and ~2e-11)
    static constexpr double tol = std::is_same_v<T, float> ? 1e-4 : 1e-9;
    //! Polar sine below which the hypot-normalized branch is used
    static constexpr double threshold
        = detail::RealVecTraits<T>::min_accurate_sintheta;
};

using RealTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(RotateTypedTest, RealTypes, );

TYPED_TEST(RotateTypedTest, axes)
{
    using ArrT = typename TestFixture::ArrT;
    for (Dbl3 const& rot : {Dbl3{1, 0, 0},
                            Dbl3{-1, 0, 0},
                            Dbl3{0, 1, 0},
                            Dbl3{0, -1, 0},
                            Dbl3{0, 0, 1},
                            Dbl3{0, 0, -1},
                            make_unit_vector(Dbl3{-1.1, 2.3, 0.9}),
                            make_unit_vector(Dbl3{1.1, -2.3, -0.9})})
    {
        SCOPED_TRACE(rot);
        check_rotate(static_array_cast<TypeParam>(rot), TestFixture::tol);
    }
    // Exactly along z, the result is just the scattered direction (+z) or the
    // scattered direction rotated by pi about y (-z)
    ArrT const dir
        = static_array_cast<TypeParam>(make_unit_vector(Dbl3{-1.1, 2.3, 0.9}));
    EXPECT_VEC_SOFT_EQ(dir, rotate(dir, ArrT{0, 0, 1}));
    EXPECT_VEC_SOFT_EQ((ArrT{-dir[X], dir[Y], -dir[Z]}),
                       rotate(dir, ArrT{0, 0, -1}));
}

TYPED_TEST(RotateTypedTest, near_z)
{
    // Polar sines straddling the threshold between the two branches (so the
    // result is continuous across it), and approaching the z axis
    double const thresh = TestFixture::threshold;
    for (double sintheta : {0.5,
                            2 * thresh,
                            1.001 * thresh,
                            0.999 * thresh,
                            0.5 * thresh,
                            1e-2 * thresh,
                            1e-4 * thresh,
                            1e-12})
    {
        for (Dbl3 const& rot : near_z_axes(sintheta))
        {
            SCOPED_TRACE(rot);
            check_rotate(static_array_cast<TypeParam>(rot), TestFixture::tol);
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
