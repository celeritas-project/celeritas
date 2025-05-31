//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/detail/SurfaceTransformer.test.cc
//---------------------------------------------------------------------------//
#include "orange/surf/detail/SurfaceTransformer.hh"

#include "corecel/Constants.hh"
#include "orange/MatrixUtils.hh"
#include "orange/surf/ConeAligned.hh"
#include "orange/surf/CylAligned.hh"
#include "orange/surf/CylCentered.hh"
#include "orange/surf/GeneralQuadric.hh"
#include "orange/surf/Plane.hh"
#include "orange/surf/PlaneAligned.hh"
#include "orange/surf/SimpleQuadric.hh"
#include "orange/surf/Sphere.hh"
#include "orange/surf/SphereCentered.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace detail
{
namespace test
{
constexpr real_type sqrt_two{constants::sqrt_two};

//---------------------------------------------------------------------------//

class SurfaceTransformerTest : public ::celeritas::test::Test
{
  protected:
    template<class... N>
    decltype(auto) array(N... args)
    {
        return Array<real_type, sizeof...(N)>{static_cast<real_type>(args)...};
    }

    // Rotate 60 degrees about X, then 30 degrees about Z, then translate
    SurfaceTransformer const transform{Transformation{
        make_rotation(
            Axis::z, Turn{1.0 / 12.0}, make_rotation(Axis::x, Turn{1.0 / 6.0})),
        {1, 2, 3}}};

    // Reflect across the YZ plane
    SurfaceTransformer const reflect{
        Transformation{make_reflection(Axis::x), Real3{0, 0, 0}}};
};

TEST_F(SurfaceTransformerTest, plane_aligned)
{
    PlaneX const orig{4};
    auto s = transform(orig);
    EXPECT_EQ(SurfaceType::p, s.surface_type());
    EXPECT_VEC_SOFT_EQ(array(0.86602540378444, 0.5, 0, 5.86602540378444),
                       s.data());

    s = reflect(orig);
    EXPECT_VEC_SOFT_EQ(array(-1, 0, 0, 4), s.data());
}

TEST_F(SurfaceTransformerTest, cyl_centered)
{
    CCylX const orig{4};
    auto s = transform(orig);
    EXPECT_EQ(SurfaceType::gq, s.surface_type());
    EXPECT_VEC_SOFT_EQ(array(0.25,
                             0.75,
                             1,
                             -0.86602540378444,
                             0,
                             0,
                             1.2320508075689,
                             -2.1339745962156,
                             -6,
                             -5.4820508075689),
                       s.data());
}

TEST_F(SurfaceTransformerTest, sphere_centered)
{
    SphereCentered orig{0.5};
    auto s = transform(orig);
    EXPECT_EQ(SurfaceType::s, s.surface_type());
    EXPECT_VEC_SOFT_EQ(array(1, 2, 3, 0.25), s.data());

    s = reflect(orig);
    EXPECT_VEC_SOFT_EQ(array(0, 0, 0, 0.25), s.data());
}

TEST_F(SurfaceTransformerTest, cyl_aligned)
{
    auto s = transform(CylY{{4, 5, -1}, 4.0});
    EXPECT_EQ(SurfaceType::gq, s.surface_type());
    EXPECT_VEC_SOFT_EQ(array(0.9375,
                             0.8125,
                             0.25,
                             0.21650635094611,
                             -0.75,
                             0.43301270189222,
                             -9.6692286340599,
                             -6.7165063509461,
                             0.56698729810778,
                             18.73172863406),
                       s.data());
}

TEST_F(SurfaceTransformerTest, plane)
{
    auto s = transform(Plane{{1 / sqrt_two, 1 / sqrt_two, 0.0}, 2 * sqrt_two});
    EXPECT_EQ(SurfaceType::p, s.surface_type());
    EXPECT_VEC_SOFT_EQ(array(0.43559574039916,
                             0.65973960844117,
                             0.61237243569579,
                             6.42061938911507),
                       s.data());
}

TEST_F(SurfaceTransformerTest, sphere)
{
    Sphere const orig{{1, 2, 3}, 0.5};

    auto s = transform(orig);
    EXPECT_EQ(SurfaceType::s, s.surface_type());
    EXPECT_VEC_SOFT_EQ(
        array(2.6650635094611, 1.1160254037844, 6.2320508075689, 0.25),
        s.data());

    s = reflect(orig);
    EXPECT_VEC_SOFT_EQ(array(-1, 2, 3, 0.25), s.data());
}

TEST_F(SurfaceTransformerTest, cone_aligned)
{
    auto s = transform(ConeX{{1, 2, 3}, 0.5});
    EXPECT_EQ(SurfaceType::gq, s.surface_type());
    EXPECT_VEC_SOFT_EQ(array(0.0625,
                             0.6875,
                             1,
                             -1.0825317547305,
                             0,
                             0,
                             0.875,
                             1.3504809471617,
                             -12.464101615138,
                             36.918906460551),
                       s.data());
}

TEST_F(SurfaceTransformerTest, simple_quadric)
{
    // See GeneralQuadric.test.cc: {3, 2, 1} radii
    auto s = transform(SimpleQuadric{{4, 9, 36}, {0, 0, 0}, -36});
    EXPECT_EQ(SurfaceType::gq, s.surface_type());
    EXPECT_VEC_SOFT_EQ(array(10.3125,
                             22.9375,
                             15.75,
                             -21.867141445557,
                             -20.25,
                             11.69134295109,
                             -11.964745962156,
                             -9.1328585544429,
                             -65.69134295109,
                             77.652245962156),
                       s.data());
}

TEST_F(SurfaceTransformerTest, general_quadric)
{
    auto s = transform(
        GeneralQuadric{{10.3125, 22.9375, 15.75},
                       {-21.867141445557, -20.25, 11.69134295109},
                       {-11.964745962156, -9.1328585544429, -65.69134295109},
                       77.652245962156});
    EXPECT_EQ(SurfaceType::gq, s.surface_type());
    EXPECT_VEC_SOFT_EQ(array(23.431849159988,
                             13.196033053329,
                             12.372117786683,
                             -24.388187891892,
                             7.61321795109,
                             -19.512634879018,
                             23.926836884239,
                             -11.904107701114,
                             -110.70146673612,
                             283.70795594936),
                       s.data());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail
}  // namespace celeritas
