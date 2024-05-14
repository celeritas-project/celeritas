//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
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
using constants::sqrt_two;

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
};

TEST_F(SurfaceTransformerTest, plane_aligned)
{
    auto s = transform(PlaneX{4.0});
    EXPECT_EQ(SurfaceType::p, s.surface_type());
    EXPECT_VEC_SOFT_EQ(array(0.86602540378444, 0.5, 0, 5.86602540378444),
                       s.data());
}

TEST_F(SurfaceTransformerTest, cyl_centered)
{
    auto s = transform(CCylX{4.0});
    EXPECT_EQ(SurfaceType::gq, s.surface_type());
    EXPECT_VEC_SOFT_EQ(array(0.041666666666667,
                             0.125,
                             0.16666666666667,
                             -0.14433756729741,
                             0,
                             0,
                             0.20534180126148,
                             -0.35566243270259,
                             -1,
                             -0.91367513459481),
                       s.data());
}

TEST_F(SurfaceTransformerTest, sphere_centered)
{
    auto s = transform(SphereCentered{0.5});
    EXPECT_EQ(SurfaceType::s, s.surface_type());
    EXPECT_VEC_SOFT_EQ(array(1, 2, 3, 0.25), s.data());
}

TEST_F(SurfaceTransformerTest, cyl_aligned)
{
    auto s = transform(CylY{{4, 5, -1}, 4.0});
    EXPECT_EQ(SurfaceType::gq, s.surface_type());
    EXPECT_VEC_SOFT_EQ(array(0.050048771168687,
                             0.043375601679528,
                             0.013346338978316,
                             0.01155826860274,
                             -0.040039016934949,
                             0.023116537205481,
                             -0.51619521203603,
                             -0.35856308203897,
                             0.030268818707785,
                             1),
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
    auto s = transform(Sphere{{1, 2, 3}, 0.5});
    EXPECT_EQ(SurfaceType::s, s.surface_type());
    EXPECT_VEC_SOFT_EQ(
        array(2.6650635094611, 1.1160254037844, 6.2320508075689, 0.25),
        s.data());
}

TEST_F(SurfaceTransformerTest, cone_aligned)
{
    auto s = transform(ConeX{{1, 2, 3}, 0.5});
    EXPECT_EQ(SurfaceType::gq, s.surface_type());
    EXPECT_VEC_SOFT_EQ(array(0.0016928995463824,
                             0.018621895010206,
                             0.027086392742118,
                             -0.029321880264446,
                             0,
                             0,
                             0.023700593649353,
                             0.036579657325568,
                             -0.33760755152529,
                             1),
                       s.data());
}

TEST_F(SurfaceTransformerTest, simple_quadric)
{
    // See GeneralQuadric.test.cc: {3, 2, 1} radii
    auto s = transform(SimpleQuadric{{4, 9, 36}, {0, 0, 0}, -36});
    EXPECT_EQ(SurfaceType::gq, s.surface_type());
    EXPECT_VEC_SOFT_EQ(array(0.13280362817871,
                             0.29538746388841,
                             0.20282735940021,
                             -0.28160346393862,
                             -0.26077803351456,
                             0.15056026784837,
                             -0.1540811320253,
                             -0.11761229107132,
                             -0.84596835722054,
                             1),
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
    EXPECT_VEC_SOFT_EQ(array(0.082591441898693,
                             0.04651273528503,
                             0.043608638838775,
                             -0.085962298132541,
                             0.026834700231138,
                             -0.068777186081102,
                             0.084336150546692,
                             -0.041959019659073,
                             -0.39019514403706,
                             1),
                       s.data());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail
}  // namespace celeritas
