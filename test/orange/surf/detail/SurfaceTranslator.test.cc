//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/detail/SurfaceTranslator.test.cc
//---------------------------------------------------------------------------//
#include "orange/surf/detail/SurfaceTranslator.hh"

#include "corecel/Constants.hh"
#include "corecel/math/SoftEqual.hh"
#include "orange/surf/detail/AllSurfaces.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace detail
{
namespace test
{
constexpr real_type sqrt_two{constants::sqrt_two};

//---------------------------------------------------------------------------//

class SurfaceTranslatorTest : public ::celeritas::test::Test
{
  protected:
    SurfaceTranslator const translate{Translation{{2, 3, 4}}};
};

TEST_F(SurfaceTranslatorTest, plane_aligned)
{
    auto px = translate(PlaneX{4.0});
    EXPECT_EQ(SurfaceType::px, px.surface_type());
    EXPECT_SOFT_EQ(6.0, px.position());
}

TEST_F(SurfaceTranslatorTest, cyl_centered)
{
    auto cx = translate(CCylX{4.0});
    EXPECT_VEC_SOFT_EQ((Real3{0, 3, 4}), cx.calc_origin());
    EXPECT_SOFT_EQ(ipow<2>(4.0), cx.radius_sq());
}

TEST_F(SurfaceTranslatorTest, sphere_centered)
{
    auto sph = translate(SphereCentered{0.5});
    EXPECT_VEC_SOFT_EQ((Real3{2, 3, 4}), sph.origin());
    EXPECT_SOFT_EQ(ipow<2>(0.5), sph.radius_sq());
}

TEST_F(SurfaceTranslatorTest, cyl_aligned)
{
    auto cy = translate(CylY{{4, 5, -1}, 4.0});
    EXPECT_VEC_SOFT_EQ((Real3{6, 0, 3}), cy.calc_origin());
    EXPECT_SOFT_EQ(ipow<2>(4.0), cy.radius_sq());
}

TEST_F(SurfaceTranslatorTest, plane)
{
    Plane const orig{{1 / sqrt_two, 1 / sqrt_two, 0.0}, 2 * sqrt_two};
    auto p = translate(orig);
    EXPECT_VEC_SOFT_EQ(orig.normal(), p.normal());
    EXPECT_SOFT_EQ(4.5 * sqrt_two, p.displacement());
}

TEST_F(SurfaceTranslatorTest, sphere)
{
    auto sph = translate(Sphere{{1, 2, 3}, 0.5});
    EXPECT_VEC_SOFT_EQ((Real3{3, 5, 7}), sph.origin());
    EXPECT_SOFT_EQ(ipow<2>(0.5), sph.radius_sq());
}

TEST_F(SurfaceTranslatorTest, cone_aligned)
{
    auto kx = translate(ConeX{{1, 2, 3}, 0.5});
    EXPECT_VEC_SOFT_EQ((Real3{3, 5, 7}), kx.origin());
    EXPECT_SOFT_EQ(ipow<2>(0.5), kx.tangent_sq());
}

TEST_F(SurfaceTranslatorTest, simple_quadric)
{
    // Ellipsoid at origin
    auto sq
        = translate(SimpleQuadric{{0.5625, 0.09, 6.25}, {0, 0, 0}, -0.5625});

    auto distances
        = sq.calc_intersections({-0.5, 3, 4}, {1, 0, 0}, SurfaceState::off);
    EXPECT_SOFT_EQ(1.5, distances[0]);
    EXPECT_SOFT_EQ(1.5 + 2.0, distances[1]);
    distances
        = sq.calc_intersections({2, 5.5, 4}, {0, -1, 0}, SurfaceState::on);
    EXPECT_SOFT_EQ(5.0, distances[0]);
    EXPECT_SOFT_EQ(no_intersection(), distances[1]);
    distances = sq.calc_intersections({2, 3, 4}, {0, 0, 1}, SurfaceState::off);
    EXPECT_SOFT_EQ(0.3, distances[1]);
    EXPECT_SOFT_EQ(no_intersection(), distances[0]);

    // Tiny ellipsoid translated in macro length scale: {0.008, 0.004, 0.005}))
    sq = translate(SimpleQuadric{{0.5, 2, 1.28}, {0, 0, 0}, -3.2e-05});
    constexpr auto coarse_eps
        = (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE ? 1e-11 : 1e-3);

    distances = sq.calc_intersections({1, 3, 4}, {1, 0, 0}, SurfaceState::off);
    EXPECT_SOFT_NEAR(1 - 0.008, distances[0], coarse_eps);
    EXPECT_SOFT_NEAR(1 + 0.008, distances[1], coarse_eps);
    distances
        = sq.calc_intersections({2, 3.004, 4}, {0, -1, 0}, SurfaceState::on);
    EXPECT_SOFT_NEAR(0.008, distances[0], coarse_eps);
    EXPECT_SOFT_EQ(no_intersection(), distances[1]);
    distances = sq.calc_intersections({2, 3, 4}, {0, 0, 1}, SurfaceState::off);
    EXPECT_SOFT_NEAR(0.005, distances[1], 10 * coarse_eps);
    EXPECT_SOFT_EQ(no_intersection(), distances[0]);
}

TEST_F(SurfaceTranslatorTest, general_quadric)
{
    // See GeneralQuadric.tst.cc
    auto gq = translate(
        GeneralQuadric{{10.3125, 22.9375, 15.75},
                       {-21.867141445557, -20.25, 11.69134295109},
                       {-11.964745962156, -9.1328585544429, -65.69134295109},
                       77.652245962156});

    Real3 const center{1 + 2, 2 + 3, 3 + 4};
    Real3 const pos{-2.464101615137754 + 2, 0 + 3, 3 + 4};
    Real3 const outward{0.8660254037844386, 0.5, 0};
    auto distances = gq.calc_intersections(center, outward, SurfaceState::off);
    EXPECT_SOFT_EQ(3.0, distances[1]);
    EXPECT_SOFT_EQ(no_intersection(), distances[0]);

    distances = gq.calc_intersections(pos, outward, SurfaceState::off);
    EXPECT_SOFT_NEAR(1.0, distances[0], SoftEqual<>{}.rel() * 10);
    EXPECT_SOFT_EQ(7.0, distances[1]);
}

TEST_F(SurfaceTranslatorTest, involute)
{
    using Real2 = Involute::Real2;
    using Sign = Chirality;
    Sign ccw = Chirality::left;
    Sign cw = Chirality::right;
    // See Involute.tst.cc
    // Cocklwise involute
    {
        auto invo = translate(Involute{{1, 0}, 2.0, 0.2, cw, 1.0, 3.0});
        EXPECT_VEC_SOFT_EQ((Real2{3, 3}), invo.origin());
        EXPECT_SOFT_EQ(2.0, invo.r_b());
        EXPECT_SOFT_EQ(0.2, invo.displacement_angle());
        EXPECT_TRUE(cw == invo.sign());
        EXPECT_SOFT_EQ(1.0, invo.tmin());
        EXPECT_SOFT_EQ(3.0, invo.tmax());
    }
    // Counterclockwise involute
    {
        auto invo = translate(Involute{{1, 0}, 2.0, 0.2, ccw, 1.0, 3.0});
        EXPECT_VEC_SOFT_EQ((Real2{3, 3}), invo.origin());
        EXPECT_SOFT_EQ(2.0, invo.r_b());
        EXPECT_SOFT_EQ(0.2, invo.displacement_angle());
        EXPECT_TRUE(ccw == invo.sign());
        EXPECT_SOFT_EQ(1.0, invo.tmin());
        EXPECT_SOFT_EQ(3.0, invo.tmax());
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail
}  // namespace celeritas
