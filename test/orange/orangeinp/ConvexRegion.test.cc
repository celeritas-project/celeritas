//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/ConvexRegion.test.cc
//---------------------------------------------------------------------------//
#include "orange/orangeinp/ConvexRegion.hh"

#include "orange/BoundingBoxUtils.hh"
#include "orange/orangeinp/ConvexSurfaceBuilder.hh"
#include "orange/orangeinp/CsgTreeUtils.hh"
#include "orange/orangeinp/detail/ConvexSurfaceState.hh"
#include "orange/orangeinp/detail/CsgUnitBuilder.hh"

#include "CsgTestUtils.hh"
#include "celeritas_test.hh"

using namespace celeritas::orangeinp::detail::test;

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
    TestResult test(ConvexRegion const& r);

  private:
    Unit unit_;
    UnitBuilder unit_builder_{&unit_, Tol::from_relative(1e-4)};
    VariantTransform transform_;
};

auto ConvexRegionTest::test(ConvexRegion const& r) -> TestResult
{
    detail::ConvexSurfaceState css;
    css.transform = &transform_;
    css.make_face_name = {};
    css.object_name = "cr";

    ConvexSurfaceBuilder insert_surface{&unit_builder_, &css};
    r.build(insert_surface);

    // Intersect the given surfaces
    auto node_id
        = unit_builder_.insert_csg(Joined{op_and, std::move(css.nodes)});

    TestResult result;
    result.node = build_infix_string(unit_.tree, node_id);
    result.surfaces = surface_strings(unit_);
    result.interior = css.local_bzone.interior;
    result.exterior = css.local_bzone.exterior;
    return result;
}

void ConvexRegionTest::TestResult::print_expected() const
{
    using std::cout;
    cout << "/***** EXPECTED REGION *****/\n"
         << "static char const expected_node[] = " << repr(this->node) << ";\n"
         << "EXPECT_EQ(expected_node, result.node);\n"
         << "static char const * const expected_surfaces[] = "
         << repr(this->surfaces) << ";\n"
         << "EXPECT_VEC_EQ(expected_surfaces, result.surfaces);\n";

    auto print_expect_req = [](char const* s, Real3 const& v) {
        cout << "EXPECT_VEC_SOFT_EQ(" << s << ", (Real3" << repr(v) << "));\n";
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
    EXPECT_EQ(expected_node, result.node);
    static char const* const expected_surfaces[] = {"Plane: x=-1",
                                                    "Plane: x=1",
                                                    "Plane: y=-2",
                                                    "Plane: y=2",
                                                    "Plane: z=-3",
                                                    "Plane: z=3"};
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

TEST_F(ConeTest, upward)
{
    auto result = this->test(Cone({1.5, 0}, 0.5));  // Lower r=1.5, height 1
    static char const expected_node[] = "all(+0, -1, -2)";
    EXPECT_EQ(expected_node, result.node);
    static char const* const expected_surfaces[]
        = {"Plane: z=-0.5", "Plane: z=0.5", "Cone z: t=1.5 at {0,0,0.5}"};
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    result.print_expected();

    EXPECT_FALSE(result.interior) << result.interior;
    EXPECT_VEC_SOFT_EQ(result.exterior.lower(), (Real3{-inf, -inf, -0.5}));
    EXPECT_VEC_SOFT_EQ(result.exterior.upper(), (Real3{inf, inf, 0.5}));
}

TEST_F(ConeTest, downward)
{
    auto result = this->test(Cone({0, 1.2}, 1.3 / 2));
    static char const expected_node[] = "all(+0, -1, -2)";
    EXPECT_EQ(expected_node, result.node);
    static char const* const expected_surfaces[] = {
        "Plane: z=-0.65", "Plane: z=0.65", "Cone z: t=0.923077 at {0,0,-0.65}"};
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    result.print_expected();
    EXPECT_FALSE(result.interior) << result.interior;
    EXPECT_VEC_SOFT_EQ(result.exterior.lower(), (Real3{-inf, -inf, -0.65}));
    EXPECT_VEC_SOFT_EQ(result.exterior.upper(), (Real3{inf, inf, 0.65}));
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
    EXPECT_EQ(expected_node, result.node);
    static char const* const expected_surfaces[]
        = {"Plane: z=-0.9", "Plane: z=0.9", "Cyl z: r=0.75"};
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    result.print_expected();
    EXPECT_VEC_SOFT_EQ(result.interior.lower(),
                       (Real3{-0.53033008588991, -0.53033008588991, -0.9}));
    EXPECT_VEC_SOFT_EQ(result.interior.upper(),
                       (Real3{0.53033008588991, 0.53033008588991, 0.9}));
    EXPECT_VEC_SOFT_EQ(result.exterior.lower(), (Real3{-0.75, -0.75, -0.9}));
    EXPECT_VEC_SOFT_EQ(result.exterior.upper(), (Real3{0.75, 0.75, 0.9}));
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
    EXPECT_EQ(expected_node, result.node);
    static char const* const expected_surfaces[]
        = {"SQuadric: {4,9,36} {0,0,0} -36"};
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    result.print_expected();

    EXPECT_FALSE(result.interior) << result.interior;
    EXPECT_VEC_SOFT_EQ(result.exterior.lower(), (Real3{-inf, -inf, -inf}));
    EXPECT_VEC_SOFT_EQ(result.exterior.upper(), (Real3{inf, inf, inf}));
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
    EXPECT_EQ(expected_node, result.node);
    static char const* const expected_surfaces[]
        = {"Plane: z=-1.2",
           "Plane: z=1.2",
           "Plane: n={0.866025,0.5,0}, d=1",
           "Plane: n={0.866025,-0.5,0}, d=-1",
           "Plane: y=-1"};
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    result.print_expected();

    EXPECT_FALSE(result.interior) << result.interior;
    EXPECT_VEC_SOFT_EQ(result.exterior.lower(), (Real3{-inf, -1, -1.2}));
    EXPECT_VEC_SOFT_EQ(result.exterior.upper(), (Real3{inf, inf, 1.2}));
}

TEST_F(PrismTest, rtriangle)
{
    auto result = this->test(Prism(3, 1.0, 1.2, 0.5));
    static char const expected_node[] = "all(+0, -1, -2, +3, -4)";
    EXPECT_EQ(expected_node, result.node);
    static char const* const expected_surfaces[]
        = {"Plane: z=-1.2",
           "Plane: z=1.2",
           "Plane: y=1",
           "Plane: n={0.866025,0.5,0}, d=-1",
           "Plane: n={0.866025,-0.5,0}, d=1"};
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    result.print_expected();

    EXPECT_FALSE(result.interior) << result.interior;
    EXPECT_VEC_SOFT_EQ(result.exterior.lower(), (Real3{-inf, -inf, -1.2}));
    EXPECT_VEC_SOFT_EQ(result.exterior.upper(), (Real3{inf, 1, 1.2}));
}

TEST_F(PrismTest, square)
{
    auto result = this->test(Prism(4, 1.0, 2.0, 0.0));
    static char const expected_node[] = "all(+0, -1, -2, -3, +4, +5)";
    EXPECT_EQ(expected_node, result.node);
    static char const* const expected_surfaces[] = {"Plane: z=-2",
                                                    "Plane: z=2",
                                                    "Plane: x=1",
                                                    "Plane: y=1",
                                                    "Plane: x=-1",
                                                    "Plane: y=-1"};
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    result.print_expected();

    EXPECT_VEC_SOFT_EQ(result.interior.lower(), (Real3{-1, -1, -2}));
    EXPECT_VEC_SOFT_EQ(result.interior.upper(), (Real3{1, 1, 2}));
    EXPECT_VEC_SOFT_EQ(result.exterior.lower(), (Real3{-1, -1, -2}));
    EXPECT_VEC_SOFT_EQ(result.exterior.upper(), (Real3{1, 1, 2}));
}

TEST_F(PrismTest, hex)
{
    auto result = this->test(Prism(6, 1.0, 2.0, 0.0));
    static char const expected_node[] = "all(+0, -1, -2, -3, +4, +5, +6, -7)";
    EXPECT_EQ(expected_node, result.node);
    static char const* const expected_surfaces[]
        = {"Plane: z=-2",
           "Plane: z=2",
           "Plane: n={0.866025,0.5,0}, d=1",
           "Plane: y=1",
           "Plane: n={0.866025,-0.5,0}, d=-1",
           "Plane: n={0.866025,0.5,0}, d=-1",
           "Plane: y=-1",
           "Plane: n={0.866025,-0.5,0}, d=1"};
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    result.print_expected();

    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    EXPECT_FALSE(result.interior) << result.interior;
    EXPECT_VEC_SOFT_EQ(result.exterior.lower(), (Real3{-inf, -1, -2}));
    EXPECT_VEC_SOFT_EQ(result.exterior.upper(), (Real3{inf, 1, 2}));
}

TEST_F(PrismTest, rhex)
{
    auto result = this->test(Prism(6, 1.0, 2.0, 0.5));
    static char const expected_node[] = "all(+0, -1, -2, -3, +4, +5, +6, -7)";
    EXPECT_EQ(expected_node, result.node);
    static char const* const expected_surfaces[]
        = {"Plane: z=-2",
           "Plane: z=2",
           "Plane: x=1",
           "Plane: n={0.5,0.866025,0}, d=1",
           "Plane: n={0.5,-0.866025,0}, d=-1",
           "Plane: x=-1",
           "Plane: n={0.5,0.866025,0}, d=-1",
           "Plane: n={0.5,-0.866025,0}, d=1"};
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    result.print_expected();

    EXPECT_FALSE(result.interior) << result.interior;
    EXPECT_VEC_SOFT_EQ(result.exterior.lower(), (Real3{-1, -inf, -2}));
    EXPECT_VEC_SOFT_EQ(result.exterior.upper(), (Real3{1, inf, 2}));
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
    EXPECT_EQ(expected_node, result.node);
    static char const* const expected_surfaces[] = {"Sphere: r=2"};
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    EXPECT_VEC_SOFT_EQ(
        result.interior.lower(),
        (Real3{-1.7320508075689, -1.7320508075689, -1.7320508075689}));
    EXPECT_VEC_SOFT_EQ(
        result.interior.upper(),
        (Real3{1.7320508075689, 1.7320508075689, 1.7320508075689}));
    EXPECT_VEC_SOFT_EQ(result.exterior.lower(), (Real3{-2, -2, -2}));
    EXPECT_VEC_SOFT_EQ(result.exterior.upper(), (Real3{2, 2, 2}));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace orangeinp
}  // namespace celeritas
