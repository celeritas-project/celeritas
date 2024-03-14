//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/ConvexRegion.test.cc
//---------------------------------------------------------------------------//
#include "orange/orangeinp/ConvexRegion.hh"

#include "orange/BoundingBoxUtils.hh"
#include "orange/MatrixUtils.hh"
#include "orange/orangeinp/ConvexSurfaceBuilder.hh"
#include "orange/orangeinp/CsgTreeUtils.hh"
#include "orange/orangeinp/detail/ConvexSurfaceState.hh"
#include "orange/orangeinp/detail/CsgUnitBuilder.hh"

#include "CsgTestUtils.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace orangeinp
{
namespace test
{
//---------------------------------------------------------------------------//
class ConvexRegionTest : public ::celeritas::test::Test
{
  private:
    using Unit = orangeinp::detail::CsgUnit;
    using UnitBuilder = orangeinp::detail::CsgUnitBuilder;
    using State = orangeinp::detail::ConvexSurfaceState;
    using Tol = UnitBuilder::Tol;

  protected:
    struct TestResult
    {
        std::string node;
        std::vector<std::string> surfaces;
        BBox interior;
        BBox exterior;

        void print_expected() const;
    };

  protected:
    TestResult test(ConvexRegionInterface const& r, VariantTransform const&);

    //! Test with no transform
    TestResult test(ConvexRegionInterface const& r)
    {
        return this->test(r, NoTransformation{});
    }

  private:
    Unit unit_;
    UnitBuilder unit_builder_{&unit_, Tol::from_relative(1e-4)};
};

//---------------------------------------------------------------------------//
auto ConvexRegionTest::test(ConvexRegionInterface const& r,
                            VariantTransform const& trans) -> TestResult
{
    detail::ConvexSurfaceState css;
    css.transform = &trans;
    css.make_face_name = {};
    css.object_name = "cr";

    ConvexSurfaceBuilder insert_surface{&unit_builder_, &css};
    r.build(insert_surface);
    EXPECT_TRUE(encloses(css.local_bzone.exterior, css.local_bzone.interior));
    EXPECT_TRUE(encloses(css.global_bzone.exterior, css.global_bzone.interior));

    // Intersect the given surfaces
    NodeId node_id
        = unit_builder_.insert_csg(Joined{op_and, std::move(css.nodes)}).first;

    TestResult result;
    result.node = build_infix_string(unit_.tree, node_id);
    result.surfaces = surface_strings(unit_);

    // Combine the bounding zones
    auto merged_bzone = calc_merged_bzone(css);
    result.interior = merged_bzone.interior;
    result.exterior = merged_bzone.exterior;

    return result;
}

//---------------------------------------------------------------------------//
void ConvexRegionTest::TestResult::print_expected() const
{
    using std::cout;
    cout << "/***** EXPECTED REGION *****/\n"
         << "static char const expected_node[] = " << repr(this->node) << ";\n"
         << "static char const * const expected_surfaces[] = "
         << repr(this->surfaces) << ";\n\n"
         << "EXPECT_EQ(expected_node, result.node);\n"
         << "EXPECT_VEC_EQ(expected_surfaces, result.surfaces);\n";

    auto print_expect_req = [](char const* s, Real3 const& v) {
        cout << "EXPECT_VEC_SOFT_EQ((Real3" << repr(v) << "), " << s << ");\n";
    };
    if (this->interior)
    {
        print_expect_req("result.interior.lower()", this->interior.lower());
        print_expect_req("result.interior.upper()", this->interior.upper());
    }
    else
    {
        cout << "EXPECT_FALSE(result.interior) << result.interior;\n";
    }
    print_expect_req("result.exterior.lower()", this->exterior.lower());
    print_expect_req("result.exterior.upper()", this->exterior.upper());
    cout << "/***************************/\n";
}

//---------------------------------------------------------------------------//
// BOX
//---------------------------------------------------------------------------//
using BoxTest = ConvexRegionTest;

TEST_F(BoxTest, errors)
{
    EXPECT_THROW(Box({-1.0, 1, 2}), RuntimeError);
    EXPECT_THROW(Box({0, 1, 2}), RuntimeError);
}

TEST_F(BoxTest, standard)
{
    auto result = this->test(Box({1, 2, 3}));
    static char const expected_node[] = "all(+0, -1, +2, -3, +4, -5)";
    static char const* const expected_surfaces[] = {"Plane: x=-1",
                                                    "Plane: x=1",
                                                    "Plane: y=-2",
                                                    "Plane: y=2",
                                                    "Plane: z=-3",
                                                    "Plane: z=3"};

    EXPECT_EQ(expected_node, result.node);
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
}

//---------------------------------------------------------------------------//
// CONE
//---------------------------------------------------------------------------//
using ConeTest = ConvexRegionTest;

TEST_F(ConeTest, errors)
{
    EXPECT_THROW(Cone({1.0, 1.0}, 0.5), RuntimeError);
    EXPECT_THROW(Cone({-1, 1}, 1), RuntimeError);
    EXPECT_THROW(Cone({0.5, 1}, 0), RuntimeError);
}

TEST_F(ConeTest, encloses)
{
    Cone const c{{1.0, 0.5}, 2.0};
    EXPECT_TRUE(c.encloses(c));
    EXPECT_TRUE(c.encloses(Cone{{0.8, 0.2}, 2.0}));
    EXPECT_TRUE(c.encloses(Cone{{0.8, 0.2}, 1.0}));
    EXPECT_FALSE(c.encloses(Cone{{0.8, 0.2}, 2.1}));
    EXPECT_FALSE(c.encloses(Cone{{0.8, 0.6}, 1.0}));
}

TEST_F(ConeTest, upward)
{
    auto result = this->test(Cone({1.5, 0}, 0.5));  // Lower r=1.5, height 1

    static char const expected_node[] = "all(+0, -1, -2)";
    static char const* const expected_surfaces[]
        = {"Plane: z=-0.5", "Plane: z=0.5", "Cone z: t=1.5 at {0,0,0.5}"};

    EXPECT_EQ(expected_node, result.node);
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    EXPECT_VEC_SOFT_EQ((Real3{-0.53033008588991, -0.53033008588991, -0.5}),
                       result.interior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{0.53033008588991, 0.53033008588991, 0}),
                       result.interior.upper());
    EXPECT_VEC_SOFT_EQ((Real3{-1.5, -1.5, -0.5}), result.exterior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{1.5, 1.5, 0.5}), result.exterior.upper());
}

TEST_F(ConeTest, downward)
{
    auto result = this->test(Cone({0, 1.2}, 1.3 / 2));

    static char const expected_node[] = "all(+0, -1, -2)";
    static char const* const expected_surfaces[] = {
        "Plane: z=-0.65", "Plane: z=0.65", "Cone z: t=0.92308 at {0,0,-0.65}"};

    EXPECT_EQ(expected_node, result.node);
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    EXPECT_VEC_SOFT_EQ((Real3{-0.42426406871193, -0.42426406871193, 0}),
                       result.interior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{0.42426406871193, 0.42426406871193, 0.65}),
                       result.interior.upper());
    EXPECT_VEC_SOFT_EQ((Real3{-1.2, -1.2, -0.65}), result.exterior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{1.2, 1.2, 0.65}), result.exterior.upper());
}

TEST_F(ConeTest, truncated)
{
    auto result = this->test(Cone({0.5, 1.5}, 0.5));

    static char const expected_node[] = "all(+0, -1, -2)";
    static char const* const expected_surfaces[]
        = {"Plane: z=-0.5", "Plane: z=0.5", "Cone z: t=1 at {0,0,-1}"};

    EXPECT_EQ(expected_node, result.node);
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    EXPECT_VEC_SOFT_EQ((Real3{-0.53033008588991, -0.53033008588991, -0.25}),
                       result.interior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{0.53033008588991, 0.53033008588991, 0.5}),
                       result.interior.upper());
    EXPECT_VEC_SOFT_EQ((Real3{-1.5, -1.5, -0.5}), result.exterior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{1.5, 1.5, 0.5}), result.exterior.upper());
}

TEST_F(ConeTest, almost_cyl)
{
    auto result = this->test(Cone({0.55, 0.45}, 10.0));

    static char const expected_node[] = "all(+0, -1, -2)";
    static char const* const expected_surfaces[]
        = {"Plane: z=-10", "Plane: z=10", "Cone z: t=0.005 at {0,0,100}"};

    EXPECT_EQ(expected_node, result.node);
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    EXPECT_VEC_SOFT_EQ((Real3{-0.31819805153395, -0.31819805153395, -10}),
                       result.interior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{0.31819805153395, 0.31819805153395, 10}),
                       result.interior.upper());
    EXPECT_VEC_SOFT_EQ((Real3{-0.55, -0.55, -10}), result.exterior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{0.55, 0.55, 10}), result.exterior.upper());
}

TEST_F(ConeTest, translated)
{
    auto result = this->test(Cone({1.0, 0.5}, 2.0), Translation{{1, 2, 3}});

    static char const expected_node[] = "all(+0, -1, -2)";
    static char const* const expected_surfaces[]
        = {"Plane: z=1", "Plane: z=5", "Cone z: t=0.125 at {1,2,9}"};

    EXPECT_EQ(expected_node, result.node);
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    EXPECT_VEC_SOFT_EQ((Real3{0.64644660940673, 1.6464466094067, 1}),
                       result.interior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{1.3535533905933, 2.3535533905933, 5}),
                       result.interior.upper());
    EXPECT_VEC_SOFT_EQ((Real3{0, 1, 1}), result.exterior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{2, 3, 5}), result.exterior.upper());
}

TEST_F(ConeTest, transformed)
{
    auto result = this->test(
        Cone({1.0, 0.5}, 2.0),
        Transformation{make_rotation(Axis::z, Turn{0.125}),  // 45deg
                       Real3{0, 0, 2.0}});

    static char const expected_node[] = "all(+0, -1, -2)";
    static char const* const expected_surfaces[]
        = {"Plane: z=0", "Plane: z=4", "Cone z: t=0.125 at {0,0,8}"};

    EXPECT_EQ(expected_node, result.node);
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    EXPECT_VEC_SOFT_EQ((Real3{-0.5, -0.5, 0}), result.interior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{0.5, 0.5, 4}), result.interior.upper());
    EXPECT_VEC_SOFT_EQ((Real3{-1.4142135623731, -1.4142135623731, 0}),
                       result.exterior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{1.4142135623731, 1.4142135623731, 4}),
                       result.exterior.upper());
}

//---------------------------------------------------------------------------//
// CYLINDER
//---------------------------------------------------------------------------//
using CylinderTest = ConvexRegionTest;

TEST_F(CylinderTest, errors)
{
    EXPECT_THROW(Cylinder(0.0, 1.0), RuntimeError);
    EXPECT_THROW(Cylinder(1.0, -1.0), RuntimeError);
}

TEST_F(CylinderTest, standard)
{
    auto result = this->test(Cylinder(0.75, 0.9));

    static char const expected_node[] = "all(+0, -1, -2)";
    static char const* const expected_surfaces[]
        = {"Plane: z=-0.9", "Plane: z=0.9", "Cyl z: r=0.75"};

    EXPECT_EQ(expected_node, result.node);
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    EXPECT_VEC_SOFT_EQ((Real3{-0.53033008588991, -0.53033008588991, -0.9}),
                       result.interior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{0.53033008588991, 0.53033008588991, 0.9}),
                       result.interior.upper());
    EXPECT_VEC_SOFT_EQ((Real3{-0.75, -0.75, -0.9}), result.exterior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{0.75, 0.75, 0.9}), result.exterior.upper());
}

TEST_F(CylinderTest, translated)
{
    auto result = this->test(Cylinder(0.75, 0.9), Translation{{1, 2, 3}});

    static char const expected_node[] = "all(+0, -1, -2)";
    static char const* const expected_surfaces[]
        = {"Plane: z=2.1", "Plane: z=3.9", "Cyl z: r=0.75 at x=1, y=2"};

    EXPECT_EQ(expected_node, result.node);
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    EXPECT_VEC_SOFT_EQ((Real3{0.46966991411009, 1.4696699141101, 2.1}),
                       result.interior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{1.5303300858899, 2.5303300858899, 3.9}),
                       result.interior.upper());
    EXPECT_VEC_SOFT_EQ((Real3{0.25, 1.25, 2.1}), result.exterior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{1.75, 2.75, 3.9}), result.exterior.upper());
}

TEST_F(CylinderTest, transformed)
{
    auto result = this->test(
        Cylinder(0.75, 0.9),
        Transformation{make_rotation(Axis::x, Turn{0.25}), Real3{0, 0, 1.0}});

    static char const expected_node[] = "all(-0, +1, -2)";
    static char const* const expected_surfaces[]
        = {"Plane: y=0.9", "Plane: y=-0.9", "Cyl y: r=0.75 at x=0, z=1"};

    EXPECT_EQ(expected_node, result.node);
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    EXPECT_VEC_SOFT_EQ((Real3{-0.53033008588991, -0.9, 0.46966991411009}),
                       result.interior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{0.53033008588991, 0.9, 1.5303300858899}),
                       result.interior.upper());
    EXPECT_VEC_SOFT_EQ((Real3{-0.75, -0.9, 0.25}), result.exterior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{0.75, 0.9, 1.75}), result.exterior.upper());
}

//---------------------------------------------------------------------------//
// ELLIPSOID
//---------------------------------------------------------------------------//
using EllipsoidTest = ConvexRegionTest;

TEST_F(EllipsoidTest, errors)
{
    EXPECT_THROW(Ellipsoid({1, 0, 2}), RuntimeError);
}

TEST_F(EllipsoidTest, standard)
{
    auto result = this->test(Ellipsoid({3, 2, 1}));

    static char const expected_node[] = "-0";
    static char const* const expected_surfaces[]
        = {"SQuadric: {4,9,36} {0,0,0} -36"};

    EXPECT_EQ(expected_node, result.node);
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    EXPECT_VEC_SOFT_EQ(
        (Real3{-1.7320508075688776, -1.1547005383792517, -0.57735026918962584}),
        result.interior.lower());
    EXPECT_VEC_SOFT_EQ(
        (Real3{1.7320508075688776, 1.1547005383792517, 0.57735026918962584}),
        result.interior.upper());
    EXPECT_VEC_SOFT_EQ((Real3{-3, -2, -1}), result.exterior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{3, 2, 1}), result.exterior.upper());
}

//---------------------------------------------------------------------------//
// INFWEDGE
//---------------------------------------------------------------------------//
using InfWedgeTest = ConvexRegionTest;

TEST_F(InfWedgeTest, errors)
{
    EXPECT_THROW(InfWedge(Turn{0}, Turn{0.51}), RuntimeError);
    EXPECT_THROW(InfWedge(Turn{0}, Turn{0}), RuntimeError);
    EXPECT_THROW(InfWedge(Turn{0}, Turn{-0.5}), RuntimeError);
    EXPECT_THROW(InfWedge(Turn{-0.1}, Turn{-0.5}), RuntimeError);
    EXPECT_THROW(InfWedge(Turn{1.1}, Turn{-0.5}), RuntimeError);
}

TEST_F(InfWedgeTest, quarter_turn)
{
    {
        SCOPED_TRACE("first quadrant");
        auto result = this->test(InfWedge(Turn{0}, Turn{0.25}));
        static char const expected_node[] = "all(+0, +1)";
        static char const* const expected_surfaces[]
            = {"Plane: y=0", "Plane: x=0"};

        EXPECT_EQ(expected_node, result.node);
        EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
        EXPECT_VEC_SOFT_EQ((Real3{0, 0, -inf}), result.interior.lower());
        EXPECT_VEC_SOFT_EQ((Real3{inf, inf, inf}), result.interior.upper());
        EXPECT_VEC_SOFT_EQ((Real3{0, 0, -inf}), result.exterior.lower());
        EXPECT_VEC_SOFT_EQ((Real3{inf, inf, inf}), result.exterior.upper());
    }
    {
        SCOPED_TRACE("second quadrant");
        auto result = this->test(InfWedge(Turn{.25}, Turn{0.25}));
        EXPECT_EQ("all(+0, -1)", result.node);
    }
    {
        SCOPED_TRACE("fourth quadrant");
        InfWedge wedge(Turn{0.75}, Turn{0.25});
        EXPECT_SOFT_EQ(0.75, wedge.start().value());
        auto result = this->test(wedge);
        EXPECT_EQ("all(+1, -0)", result.node);
    }
    {
        SCOPED_TRACE("north quadrant");
        auto result = this->test(InfWedge(Turn{0.125}, Turn{0.25}));
        EXPECT_EQ("all(-2, +3)", result.node);
    }
    {
        SCOPED_TRACE("east quadrant");
        auto result = this->test(InfWedge(Turn{1 - 0.125}, Turn{0.25}));
        EXPECT_EQ("all(+2, +3)", result.node);
        static char const expected_node[] = "all(+2, +3)";
        EXPECT_EQ(expected_node, result.node);
        EXPECT_FALSE(result.interior) << result.interior;
        EXPECT_EQ(BBox::from_infinite(), result.exterior);
    }
    {
        SCOPED_TRACE("west quadrant");
        auto result = this->test(InfWedge(Turn{0.375}, Turn{0.25}));
        static char const expected_node[] = "all(-2, -3)";
        static char const* const expected_surfaces[]
            = {"Plane: y=0",
               "Plane: x=0",
               "Plane: n={0.70711,-0.70711,0}, d=0",
               "Plane: n={0.70711,0.70711,0}, d=0",
               "Plane: n={0.70711,0.70711,0}, d=0",
               "Plane: n={0.70711,-0.70711,0}, d=0"};

        EXPECT_EQ(expected_node, result.node);
        EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    }
}

TEST_F(InfWedgeTest, half_turn)
{
    {
        SCOPED_TRACE("north half");
        auto result = this->test(InfWedge(Turn{0}, Turn{0.5}));
        EXPECT_EQ("+0", result.node);
        EXPECT_VEC_SOFT_EQ((Real3{-inf, 0, -inf}), result.interior.lower());
        EXPECT_VEC_SOFT_EQ((Real3{inf, inf, inf}), result.interior.upper());
        EXPECT_VEC_SOFT_EQ((Real3{-inf, 0, -inf}), result.exterior.lower());
        EXPECT_VEC_SOFT_EQ((Real3{inf, inf, inf}), result.exterior.upper());
    }
    {
        SCOPED_TRACE("south half");
        auto result = this->test(InfWedge(Turn{0.5}, Turn{0.5}));
        EXPECT_EQ("-0", result.node);
    }
    {
        SCOPED_TRACE("northeast half");
        auto result = this->test(InfWedge(Turn{0.125}, Turn{0.5}));
        static char const expected_node[] = "-1";
        static char const* const expected_surfaces[]
            = {"Plane: y=0",
               "Plane: n={0.70711,-0.70711,0}, d=0",
               "Plane: n={0.70711,-0.70711,0}, d=0"};

        EXPECT_EQ(expected_node, result.node);
        EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    }
}

//---------------------------------------------------------------------------//
// PRISM
//---------------------------------------------------------------------------//
using PrismTest = ConvexRegionTest;

TEST_F(PrismTest, errors)
{
    EXPECT_THROW(Prism(2, 1.0, 1.0, 0.0), RuntimeError);  // sides
    EXPECT_THROW(Prism(5, 1.0, 0.0, 0.5), RuntimeError);  // height
    EXPECT_THROW(Prism(5, 1.0, 1.0, 1.0), RuntimeError);  // orientation
}

TEST_F(PrismTest, triangle)
{
    auto result = this->test(Prism(3, 1.0, 1.2, 0.0));

    static char const expected_node[] = "all(+0, -1, -2, +3, +4)";
    static char const* const expected_surfaces[] = {"Plane: z=-1.2",
                                                    "Plane: z=1.2",
                                                    "Plane: "
                                                    "n={0.86603,0.5,0}, d=1",
                                                    "Plane: "
                                                    "n={0.86603,-0.5,0}, d=-1",
                                                    "Plane: y=-1"};

    EXPECT_EQ(expected_node, result.node);
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    EXPECT_VEC_SOFT_EQ((Real3{-1, -1, -1.2}), result.interior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{1, 1, 1.2}), result.interior.upper());
    EXPECT_VEC_SOFT_EQ((Real3{-2, -1, -1.2}), result.exterior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{2, 2, 1.2}), result.exterior.upper());
}

TEST_F(PrismTest, rtriangle)
{
    auto result = this->test(Prism(3, 1.0, 1.2, 0.5));

    static char const expected_node[] = "all(+0, -1, -2, +3, -4)";
    static char const* const expected_surfaces[] = {"Plane: z=-1.2",
                                                    "Plane: z=1.2",
                                                    "Plane: y=1",
                                                    "Plane: "
                                                    "n={0.86603,0.5,0}, d=-1",
                                                    "Plane: "
                                                    "n={0.86603,-0.5,0}, d=1"};

    EXPECT_EQ(expected_node, result.node);
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    EXPECT_VEC_SOFT_EQ((Real3{-1, -1, -1.2}), result.interior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{1, 1, 1.2}), result.interior.upper());
    EXPECT_VEC_SOFT_EQ((Real3{-2, -2, -1.2}), result.exterior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{2, 1, 1.2}), result.exterior.upper());
}

TEST_F(PrismTest, square)
{
    auto result = this->test(Prism(4, 1.0, 2.0, 0.0));

    static char const expected_node[] = "all(+0, -1, -2, -3, +4, +5)";
    static char const* const expected_surfaces[] = {"Plane: z=-2",
                                                    "Plane: z=2",
                                                    "Plane: x=1",
                                                    "Plane: y=1",
                                                    "Plane: x=-1",
                                                    "Plane: y=-1"};

    EXPECT_EQ(expected_node, result.node);
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    EXPECT_VEC_SOFT_EQ((Real3{-1, -1, -2}), result.interior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{1, 1, 2}), result.interior.upper());
    EXPECT_VEC_SOFT_EQ((Real3{-1, -1, -2}), result.exterior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{1, 1, 2}), result.exterior.upper());
}

TEST_F(PrismTest, hex)
{
    auto result = this->test(Prism(6, 1.0, 2.0, 0.0));

    static char const expected_node[] = "all(+0, -1, -2, -3, +4, +5, +6, -7)";
    static char const* const expected_surfaces[]
        = {"Plane: z=-2",
           "Plane: z=2",
           "Plane: n={0.86603,0.5,0}, d=1",
           "Plane: y=1",
           "Plane: n={0.86603,-0.5,0}, d=-1",
           "Plane: n={0.86603,0.5,0}, d=-1",
           "Plane: y=-1",
           "Plane: n={0.86603,-0.5,0}, d=1"};

    EXPECT_EQ(expected_node, result.node);
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    EXPECT_VEC_SOFT_EQ((Real3{-1, -1, -2}), result.interior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{1, 1, 2}), result.interior.upper());
    EXPECT_VEC_SOFT_EQ((Real3{-1.1547005383793, -1, -2}),
                       result.exterior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{1.1547005383793, 1, 2}), result.exterior.upper());
}

TEST_F(PrismTest, rhex)
{
    auto result = this->test(Prism(6, 1.0, 2.0, 0.5));

    static char const expected_node[] = "all(+0, -1, -2, -3, +4, +5, +6, -7)";
    static char const* const expected_surfaces[]
        = {"Plane: z=-2",
           "Plane: z=2",
           "Plane: x=1",
           "Plane: n={0.5,0.86603,0}, d=1",
           "Plane: n={0.5,-0.86603,0}, d=-1",
           "Plane: x=-1",
           "Plane: n={0.5,0.86603,0}, d=-1",
           "Plane: n={0.5,-0.86603,0}, d=1"};

    EXPECT_EQ(expected_node, result.node);
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    EXPECT_VEC_SOFT_EQ((Real3{-1, -1, -2}), result.interior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{1, 1, 2}), result.interior.upper());
    EXPECT_VEC_SOFT_EQ((Real3{-1, -1.1547005383793, -2}),
                       result.exterior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{1, 1.1547005383793, 2}), result.exterior.upper());
}

//---------------------------------------------------------------------------//
// SPHERE
//---------------------------------------------------------------------------//
using SphereTest = ConvexRegionTest;

TEST_F(SphereTest, errors)
{
    EXPECT_THROW(Sphere(-1), RuntimeError);
}

TEST_F(SphereTest, standard)
{
    auto result = this->test(Sphere(2.0));

    static char const expected_node[] = "-0";
    static char const* const expected_surfaces[] = {"Sphere: r=2"};

    EXPECT_EQ(expected_node, result.node);
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    EXPECT_VEC_SOFT_EQ(
        (Real3{-1.7320508075689, -1.7320508075689, -1.7320508075689}),
        result.interior.lower());
    EXPECT_VEC_SOFT_EQ(
        (Real3{1.7320508075689, 1.7320508075689, 1.7320508075689}),
        result.interior.upper());
    EXPECT_VEC_SOFT_EQ((Real3{-2, -2, -2}), result.exterior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{2, 2, 2}), result.exterior.upper());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace orangeinp
}  // namespace celeritas
