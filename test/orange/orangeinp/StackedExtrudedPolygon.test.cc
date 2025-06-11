//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/StackedExtrudedPolygon.test.cc
//---------------------------------------------------------------------------//
#include "orange/orangeinp/StackedExtrudedPolygon.hh"

#include "orange/orangeinp/Shape.hh"
#include "orange/orangeinp/Solid.hh"
#include "orange/orangeinp/Transformed.hh"

#include "CsgTestUtils.hh"
#include "ObjectTestBase.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace orangeinp
{
namespace test
{

//---------------------------------------------------------------------------//
class StackedExtrudedPolygonTest : public ObjectTestBase
{
  protected:
    using VecReal = StackedExtrudedPolygon::VecReal;
    using VecReal2 = StackedExtrudedPolygon::VecReal2;
    using VecReal3 = StackedExtrudedPolygon::VecReal3;

    Tol tolerance() const override { return Tol::from_default(); }
};

//---------------------------------------------------------------------------//
/*
 * Test a convex polygon extruded along two segments, with scaling at a 45
 * degree angle.
 */
TEST_F(StackedExtrudedPolygonTest, scaled_convex_stack)
{
    VecReal2 polygon{{-1, -1}, {-1, 1}, {1, 1}, {1, -1}};
    VecReal3 polyline{{0, 0, 0}, {0, 0, 1}, {0, 0, 1.5}};
    VecReal scaling{1, 1, 0.5};

    this->build_volume(StackedExtrudedPolygon{
        "pc", std::move(polygon), std::move(polyline), std::move(scaling)});

    static char const* const expected_surface_strings[]
        = {"Plane: z=0",
           "Plane: z=1",
           "Plane: x=-1",
           "Plane: y=1",
           "Plane: x=1",
           "Plane: y=-1",
           "Plane: z=1.5",
           "Plane: n={0.70711,0,-0.70711}, d=-1.4142",
           "Plane: n={-0,0.70711,0.70711}, d=1.4142",
           "Plane: n={0.70711,0,0.70711}, d=1.4142",
           "Plane: n={0,0.70711,-0.70711}, d=-1.4142"};

    static char const* const expected_volume_strings[]
        = {"any(all(+0, -1, +2, -3, -4, +5), all(+1, -6, +7, -8, -9, +10))"};
    static char const* const expected_md_strings[] = {"",
                                                      "",
                                                      "pc@0.0.0.mz",
                                                      ("pc@0.0.0.pz,pc@0.0.1."
                                                       "mz"),
                                                      "",
                                                      "pc@0.0.0.p0",
                                                      "pc@0.0.0.p1",
                                                      "",
                                                      "pc@0.0.0.p2",
                                                      "",
                                                      "pc@0.0.0.p3",
                                                      "pc@0.0.0",
                                                      "pc@0.0.1.pz",
                                                      "",
                                                      "pc@0.0.1.p0",
                                                      "pc@0.0.1.p1",
                                                      "",
                                                      "pc@0.0.1.p2",
                                                      "",
                                                      "pc@0.0.1.p3",
                                                      "pc@0.0.1",
                                                      "pc@0.0"};

    auto const& u = this->unit();
    EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
    EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
    EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
}

//---------------------------------------------------------------------------//
/*
 * Test a convex polygon extruded along two segments, with the second segment
 * bending at a 45 degree angle.
 */
TEST_F(StackedExtrudedPolygonTest, skewed_convex_stack)
{
    VecReal2 polygon{{-1, -1}, {-1, 1}, {1, 1}, {1, -1}};
    VecReal3 polyline{{0, 0, 0}, {0, 0, 1}, {1, 1, 2}};
    VecReal scaling{1, 1, 1};

    this->build_volume(StackedExtrudedPolygon{
        "pc", std::move(polygon), std::move(polyline), std::move(scaling)});

    static char const* const expected_surface_strings[]
        = {"Plane: z=0",
           "Plane: z=1",
           "Plane: x=-1",
           "Plane: y=1",
           "Plane: x=1",
           "Plane: y=-1",
           "Plane: z=2",
           "Plane: n={0.70711,0,-0.70711}, d=-1.4142",
           "Plane: n={0,0.70711,-0.70711}, d=0",
           "Plane: n={0.70711,0,-0.70711}, d=0",
           "Plane: n={0,0.70711,-0.70711}, d=-1.4142"};

    static char const* const expected_volume_strings[]
        = {"any(all(+0, -1, +2, -3, -4, +5), all(+1, -6, +7, -8, -9, +10))"};
    static char const* const expected_md_strings[] = {"",
                                                      "",
                                                      "pc@0.0.0.mz",
                                                      ("pc@0.0.0.pz,pc@0.0.1."
                                                       "mz"),
                                                      "",
                                                      "pc@0.0.0.p0",
                                                      "pc@0.0.0.p1",
                                                      "",
                                                      "pc@0.0.0.p2",
                                                      "",
                                                      "pc@0.0.0.p3",
                                                      "pc@0.0.0",
                                                      "pc@0.0.1.pz",
                                                      "",
                                                      "pc@0.0.1.p0",
                                                      "pc@0.0.1.p1",
                                                      "",
                                                      "pc@0.0.1.p2",
                                                      "",
                                                      "pc@0.0.1.p3",
                                                      "pc@0.0.1",
                                                      "pc@0.0"};

    auto const& u = this->unit();
    EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
    EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
    EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
}

//---------------------------------------------------------------------------//
/*
 * Test a polygon with two "levels" of concavity, extruded along a single
 * segment.
 *
 *  3 ______ . . . . . . .  ____
 *     |    |              |    |
 *  2 _|    |     ____     |    |
 *     |    |    |    |    |    |
 *  1 _|    |____|. . |____|    |
 *     |                        |
 *  0 _|________________________|
 *     |    |    |    |    |    |
 *     0    1    2    3    4    5
 */
TEST_F(StackedExtrudedPolygonTest, concave_stack)
{
    VecReal2 polygon{{0, 0},
                     {0, 3},
                     {1, 3},
                     {1, 1},
                     {2, 1},
                     {2, 2},
                     {3, 2},
                     {3, 1},
                     {4, 1},
                     {4, 3},
                     {5, 3},
                     {5, 0}};
    VecReal3 polyline{{0, 0, 0}, {0, 0, 1}};
    VecReal scaling{1, 1};

    this->build_volume(StackedExtrudedPolygon{
        "pc", std::move(polygon), std::move(polyline), std::move(scaling)});

    static char const* const expected_surface_strings[] = {
        "Plane: z=0",
        "Plane: z=1",
        "Plane: x=0",
        "Plane: y=3",
        "Plane: x=5",
        "Plane: y=0",
        "Plane: x=4",
        "Plane: y=1",
        "Plane: x=1",
        "Plane: x=2",
        "Plane: y=2",
        "Plane: x=3",
    };

    static char const* const expected_volume_strings[] = {
        R"(all(+0, -1, +2, -3, -4, +5, !all(+0, -1, -3, -6, +7, +8, !all(+0, -1, +7, +9, -10, -11))))"};

    static char const* const expected_md_strings[]
        = {"",
           "",
           "pc@0.0.0.mz,pc@1.0.0.mz,pc@2.0.0.mz",
           "pc@0.0.0.pz,pc@1.0.0.pz,pc@2.0.0.pz",
           "",
           "pc@0.0.0.p0",
           "pc@0.0.0.p1,pc@1.0.0.p3",
           "",
           "pc@0.0.0.p2",
           "",
           "pc@0.0.0.p3",
           "pc@0.0,pc@0.0.0",
           "pc@1.0.0.p0",
           "",
           "pc@1.0.0.p1,pc@2.0.0.p3",
           "pc@1.0.0.p2",
           "pc@1.0,pc@1.0.0",
           "pc@2.0.0.p0",
           "pc@2.0.0.p1",
           "",
           "pc@2.0.0.p2",
           "",
           "pc@1.cu,pc@2.0,pc@2.0.0",
           "pc@1.ncu",
           "pc@0.cu,pc@1.d",
           "pc@0.ncu",
           "pc@0.d"};

    auto const& u = this->unit();
    EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
    EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
    EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace orangeinp
}  // namespace celeritas
