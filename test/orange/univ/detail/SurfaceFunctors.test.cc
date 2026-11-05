//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/univ/detail/SurfaceFunctors.test.cc
//---------------------------------------------------------------------------//
#include "orange/univ/detail/SurfaceFunctors.hh"

#include <array>
#include <cmath>

#include "orange/surf/PlaneAligned.hh"
#include "orange/surf/Sphere.hh"
#include "orange/surf/SurfaceTypeTraits.hh"
#include "orange/surf/detail/AllSurfaces.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace detail
{
namespace test
{
//---------------------------------------------------------------------------//
constexpr std::array<SurfaceType, 18> supported_surface_types{SurfaceType::px,
                                                              SurfaceType::py,
                                                              SurfaceType::pz,
                                                              SurfaceType::p,
                                                              SurfaceType::cxc,
                                                              SurfaceType::cyc,
                                                              SurfaceType::czc,
                                                              SurfaceType::sc,
                                                              SurfaceType::cx,
                                                              SurfaceType::cy,
                                                              SurfaceType::cz,
                                                              SurfaceType::s,
                                                              SurfaceType::kx,
                                                              SurfaceType::ky,
                                                              SurfaceType::kz,
                                                              SurfaceType::sq,
                                                              SurfaceType::gq,
                                                              SurfaceType::tzc};

constexpr std::array<SurfaceTypeGroup, 18> expected_surface_groups{
    SurfaceTypeGroup::planar,
    SurfaceTypeGroup::planar,
    SurfaceTypeGroup::planar,
    SurfaceTypeGroup::planar,
    SurfaceTypeGroup::quadric,
    SurfaceTypeGroup::quadric,
    SurfaceTypeGroup::quadric,
    SurfaceTypeGroup::quadric,
    SurfaceTypeGroup::quadric,
    SurfaceTypeGroup::quadric,
    SurfaceTypeGroup::quadric,
    SurfaceTypeGroup::quadric,
    SurfaceTypeGroup::quadric,
    SurfaceTypeGroup::quadric,
    SurfaceTypeGroup::quadric,
    SurfaceTypeGroup::quadric,
    SurfaceTypeGroup::quadric,
    SurfaceTypeGroup::quartic};

constexpr bool surface_type_groups_are_valid()
{
    for (size_type i = 0; i < supported_surface_types.size(); ++i)
    {
        if (surface_type_group(supported_surface_types[i])
            != expected_surface_groups[i])
        {
            return false;
        }
    }
    return true;
}

static_assert(surface_type_groups_are_valid());
static_assert(surface_type_group(PlaneX::surface_type())
              == SurfaceTypeGroup::planar);
static_assert(surface_type_group(PlaneY::surface_type())
              == SurfaceTypeGroup::planar);
static_assert(surface_type_group(PlaneZ::surface_type())
              == SurfaceTypeGroup::planar);
static_assert(surface_type_group(CylCentered<Axis::x>::surface_type())
              == SurfaceTypeGroup::quadric);
static_assert(surface_type_group(CylCentered<Axis::y>::surface_type())
              == SurfaceTypeGroup::quadric);
static_assert(surface_type_group(CylCentered<Axis::z>::surface_type())
              == SurfaceTypeGroup::quadric);
static_assert(surface_type_group(CylAligned<Axis::x>::surface_type())
              == SurfaceTypeGroup::quadric);
static_assert(surface_type_group(CylAligned<Axis::y>::surface_type())
              == SurfaceTypeGroup::quadric);
static_assert(surface_type_group(CylAligned<Axis::z>::surface_type())
              == SurfaceTypeGroup::quadric);
static_assert(surface_type_group(ConeAligned<Axis::x>::surface_type())
              == SurfaceTypeGroup::quadric);
static_assert(surface_type_group(ConeAligned<Axis::y>::surface_type())
              == SurfaceTypeGroup::quadric);
static_assert(surface_type_group(ConeAligned<Axis::z>::surface_type())
              == SurfaceTypeGroup::quadric);

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class SurfaceFunctorsTest : public ::celeritas::test::Test
{
  protected:
    PlaneX px_{1.25};
    Sphere s_{{2.25, 1, 0}, 1.25};
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST(SurfaceTypeGroupTest, mappings)
{
    for (size_type i = 0; i < supported_surface_types.size(); ++i)
    {
        EXPECT_EQ(expected_surface_groups[i],
                  surface_type_group(supported_surface_types[i]));
    }

    EXPECT_EQ(SurfaceTypeGroup::planar,
              surface_type_group(PlaneX::surface_type()));
    EXPECT_EQ(SurfaceTypeGroup::planar,
              surface_type_group(PlaneY::surface_type()));
    EXPECT_EQ(SurfaceTypeGroup::planar,
              surface_type_group(PlaneZ::surface_type()));
    EXPECT_EQ(SurfaceTypeGroup::quadric,
              surface_type_group(CylCentered<Axis::x>::surface_type()));
    EXPECT_EQ(SurfaceTypeGroup::quadric,
              surface_type_group(CylCentered<Axis::y>::surface_type()));
    EXPECT_EQ(SurfaceTypeGroup::quadric,
              surface_type_group(CylCentered<Axis::z>::surface_type()));
    EXPECT_EQ(SurfaceTypeGroup::quadric,
              surface_type_group(CylAligned<Axis::x>::surface_type()));
    EXPECT_EQ(SurfaceTypeGroup::quadric,
              surface_type_group(CylAligned<Axis::y>::surface_type()));
    EXPECT_EQ(SurfaceTypeGroup::quadric,
              surface_type_group(CylAligned<Axis::z>::surface_type()));
    EXPECT_EQ(SurfaceTypeGroup::quadric,
              surface_type_group(ConeAligned<Axis::x>::surface_type()));
    EXPECT_EQ(SurfaceTypeGroup::quadric,
              surface_type_group(ConeAligned<Axis::y>::surface_type()));
    EXPECT_EQ(SurfaceTypeGroup::quadric,
              surface_type_group(ConeAligned<Axis::z>::surface_type()));
}

TEST_F(SurfaceFunctorsTest, num_intersections)
{
    EXPECT_EQ(1, visit_surface_type(NumIntersections{}, SurfaceType::px));
    EXPECT_EQ(2, visit_surface_type(NumIntersections{}, SurfaceType::sc));
}

//---------------------------------------------------------------------------//

TEST_F(SurfaceFunctorsTest, calc_safety_distance)
{
    Real3 pos;
    CalcSafetyDistance calc_distance{pos};

    real_type eps = 1e-4;
    pos = {1.25_r + eps, 1, 0};
    EXPECT_SOFT_EQ(eps, calc_distance(px_));
    EXPECT_SOFT_EQ(0.25 + eps, calc_distance(s_));

    pos = {1.25_r, 1, 0};
    EXPECT_SOFT_EQ(0, calc_distance(px_));
    EXPECT_SOFT_EQ(0.25, calc_distance(s_));

    pos = {1.25_r - eps, 1, 0};
    EXPECT_SOFT_EQ(eps, calc_distance(px_));
    EXPECT_SOFT_EQ(0.25 - eps, calc_distance(s_));

    pos = {1_r - eps, 1, 0};
    EXPECT_SOFT_EQ(0.25 + eps, calc_distance(px_));
    EXPECT_SOFT_EQ(eps, calc_distance(s_));

    pos = {3.5_r + eps, 1, 0};
    EXPECT_SOFT_EQ(2.25 + eps, calc_distance(px_));
    EXPECT_SOFT_NEAR(0.0 + eps, calc_distance(s_), coarse_eps);

    pos = {3.5_r, 1, 0};
    EXPECT_SOFT_EQ(2.25, calc_distance(px_));
    EXPECT_SOFT_EQ(0.0, calc_distance(s_));
}

//---------------------------------------------------------------------------//

TEST_F(SurfaceFunctorsTest, calc_intersections)
{
    Array<FaceId, 8> face;
    Array<real_type, 8> distance;
    Array<size_type, 8> isect;
    isect.fill(99);
    TempNextFace next_face{
        face.data(), distance.data(), isect.data(), face.size()};

    Real3 const pos{-7, 0, 0};
    Real3 const dir{1, 0, 0};
    PlaneX on_face_plane{-7};
    SphereCentered unit_sphere{1};
    Sphere tangent_sphere{Real3{0, 1, 0}, 1};
    Sphere no_hit_sphere{Real3{0, 3, 0}, 0.5};

#if CELERITAS_ORANGE_TORUS
    Toroid torus{Real3{0, 0, 0}, 5, 1, 1};
#endif

    CalcIntersections calc{13, pos, dir, FaceId{0}, false, next_face};
    calc(on_face_plane);
    calc(unit_sphere);
    calc(tangent_sphere);
#if CELERITAS_ORANGE_TORUS
    calc(torus);
#endif
    calc(no_hit_sphere);

#if CELERITAS_ORANGE_TORUS
    EXPECT_EQ(5, calc.face_idx());
    ASSERT_EQ(7, calc.isect_idx());
    EXPECT_EQ(FaceId{1}, face[0]);
    EXPECT_EQ(FaceId{1}, face[1]);
    EXPECT_EQ(FaceId{2}, face[2]);
    EXPECT_EQ(FaceId{3}, face[3]);
    EXPECT_EQ(FaceId{3}, face[4]);
    EXPECT_EQ(FaceId{3}, face[5]);
    EXPECT_EQ(FaceId{3}, face[6]);
    EXPECT_SOFT_EQ(6, distance[0]);
    EXPECT_SOFT_EQ(8, distance[1]);
    EXPECT_SOFT_EQ(7, distance[2]);
    EXPECT_SOFT_EQ(13, distance[3]);
    EXPECT_SOFT_EQ(1, distance[4]);
    EXPECT_SOFT_EQ(11, distance[5]);
    EXPECT_SOFT_EQ(3, distance[6]);
    for (size_type i = 0; i < calc.isect_idx(); ++i)
    {
        EXPECT_EQ(i, isect[i]);
    }
#else
    EXPECT_EQ(4, calc.face_idx());
    ASSERT_EQ(3, calc.isect_idx());
    EXPECT_SOFT_EQ(6, distance[0]);
    EXPECT_SOFT_EQ(8, distance[1]);
    EXPECT_SOFT_EQ(7, distance[2]);
#endif

    Array<size_type, 8> simple_isect;
    simple_isect.fill(99);
    TempNextFace simple_next{
        face.data(), distance.data(), simple_isect.data(), face.size()};
    CalcIntersections simple{13, pos, dir, FaceId{0}, true, simple_next};
    simple(on_face_plane);
    simple(unit_sphere);
    simple(tangent_sphere);
#if CELERITAS_ORANGE_TORUS
    simple(torus);
#endif
    simple(no_hit_sphere);
    EXPECT_EQ(calc.isect_idx(), simple.isect_idx());
    for (size_type i = 0; i < simple_isect.size(); ++i)
    {
        EXPECT_EQ(size_type{99}, simple_isect[i]);
    }
}

//---------------------------------------------------------------------------//

TEST_F(SurfaceFunctorsTest, on_surface_intersections)
{
    Array<FaceId, 4> face;
    Array<real_type, 4> distance;
    Array<size_type, 4> isect;
    TempNextFace next_face{
        face.data(), distance.data(), isect.data(), face.size()};

    CalcIntersections sphere_calc{
        10, Real3{1, 0, 0}, Real3{-1, 0, 0}, FaceId{0}, false, next_face};
    sphere_calc(SphereCentered{1});
    ASSERT_EQ(1, sphere_calc.isect_idx());
    EXPECT_SOFT_EQ(2, distance[0]);
    EXPECT_EQ(no_intersection(), distance[1]);

#if CELERITAS_ORANGE_TORUS
    Toroid torus{Real3{0, 0, 0}, 5, 1, 1};
    CalcIntersections torus_calc{
        20, Real3{6, 0, 0}, Real3{-1, 0, 0}, FaceId{0}, false, next_face};
    torus_calc(torus);
    ASSERT_EQ(3, torus_calc.isect_idx());
    EXPECT_SOFT_EQ(2, distance[0]);
    EXPECT_SOFT_EQ(12, distance[1]);
    EXPECT_SOFT_EQ(10, distance[2]);
    EXPECT_EQ(no_intersection(), distance[3]);
#endif
}

//---------------------------------------------------------------------------//

TEST_F(SurfaceFunctorsTest, max_distance)
{
#if CELERITAS_ORANGE_TORUS
    Toroid torus{Real3{0, 0, 0}, 5, 1, 1};
    Real3 const pos{-7, 0, 0};
    Real3 const dir{1, 0, 0};

    Array<FaceId, 4> face;
    Array<real_type, 4> distance;
    Array<size_type, 4> isect;
    TempNextFace next_face{
        face.data(), distance.data(), isect.data(), face.size()};

    CalcIntersections equal_cutoff{3, pos, dir, FaceId{}, false, next_face};
    equal_cutoff(torus);
    ASSERT_EQ(2, equal_cutoff.isect_idx());
    EXPECT_SOFT_EQ(1, distance[0]);
    EXPECT_SOFT_EQ(3, distance[1]);

    CalcIntersections below_cutoff{
        std::nextafter(3.0, 0.0), pos, dir, FaceId{}, false, next_face};
    below_cutoff(torus);
    ASSERT_EQ(1, below_cutoff.isect_idx());
    EXPECT_SOFT_EQ(1, distance[0]);
#endif
}

//---------------------------------------------------------------------------//

TEST_F(SurfaceFunctorsTest, cylinder_near_axis)
{
    CCylX cyl{1};
    auto dist = cyl.calc_intersections(
        Real3{0, 2, 0}, Real3{1, -1e-6, 0}, SurfaceState::off);
    EXPECT_SOFT_EQ(750000, dist[0]);
    EXPECT_EQ(no_intersection(), dist[1]);

    dist = cyl.calc_intersections(
        Real3{0, 1, 0}, Real3{1, -1e-6, 0}, SurfaceState::on);
    EXPECT_EQ(no_intersection(), dist[0]);
    EXPECT_EQ(no_intersection(), dist[1]);

    real_type const threshold = ipow<2>(Tolerance<>::sqrt_quadratic());
    auto at_threshold
        = detail::solve_intersections(detail::QuadraticIntersectionInput{
            threshold, -1e-5, 0, SurfaceState::on});
    EXPECT_SOFT_EQ(2e5, at_threshold[0]);
    EXPECT_EQ(no_intersection(), at_threshold[1]);

    auto below_threshold
        = detail::solve_intersections(detail::QuadraticIntersectionInput{
            std::nextafter(threshold, 0.0), -1e-5, 0, SurfaceState::on});
    EXPECT_EQ(no_intersection(), below_threshold[0]);
    EXPECT_EQ(no_intersection(), below_threshold[1]);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail
}  // namespace celeritas
