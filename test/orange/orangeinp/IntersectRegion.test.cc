//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/IntersectRegion.test.cc
//---------------------------------------------------------------------------//
#include "orange/orangeinp/IntersectRegion.hh"

#include "orange/BoundingBoxUtils.hh"
#include "orange/MatrixUtils.hh"
#include "orange/orangeinp/CsgTreeUtils.hh"
#include "orange/orangeinp/IntersectSurfaceBuilder.hh"
#include "orange/orangeinp/detail/CsgUnitBuilder.hh"
#include "orange/orangeinp/detail/IntersectSurfaceState.hh"
#include "orange/orangeinp/detail/SenseEvaluator.hh"

#include "CsgTestUtils.hh"
#include "celeritas_test.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
std::ostream& operator<<(std::ostream& os, SignedSense s)
{
    return (os << to_cstring(s));
}

namespace orangeinp
{
namespace test
{
//---------------------------------------------------------------------------//
class IntersectRegionTest : public ::celeritas::test::Test
{
  private:
    using Unit = orangeinp::detail::CsgUnit;
    using UnitBuilder = orangeinp::detail::CsgUnitBuilder;
    using State = orangeinp::detail::IntersectSurfaceState;
    using Tol = UnitBuilder::Tol;

  protected:
    struct TestResult
    {
        std::string node;
        std::vector<std::string> surfaces;
        BBox interior;
        BBox exterior;
        NodeId node_id;

        void print_expected() const;
    };

  protected:
    TestResult test(IntersectRegionInterface const& r, VariantTransform const&);

    //! Test with no transform
    TestResult test(IntersectRegionInterface const& r)
    {
        return this->test(r, NoTransformation{});
    }

    SignedSense calc_sense(NodeId n, Real3 const& pos)
    {
        CELER_EXPECT(n < unit_.tree.size());
        detail::SenseEvaluator eval_sense(unit_.tree, unit_.surfaces, pos);
        return eval_sense(n);
    }

    Unit const& unit() const { return unit_; }

  private:
    Unit unit_;
    UnitBuilder unit_builder_{
        &unit_, Tol::from_relative(1e-4), BBox::from_infinite()};
};

//---------------------------------------------------------------------------//
auto IntersectRegionTest::test(IntersectRegionInterface const& r,
                               VariantTransform const& trans) -> TestResult
{
    detail::IntersectSurfaceState css;
    css.transform = &trans;
    css.make_face_name = {};
    css.object_name = "cr";

    IntersectSurfaceBuilder insert_surface{&unit_builder_, &css};
    r.build(insert_surface);
    if (css.local_bzone.exterior || css.local_bzone.interior)
    {
        EXPECT_TRUE(
            encloses(css.local_bzone.exterior, css.local_bzone.interior));
    }
    if (css.global_bzone.exterior || css.global_bzone.interior)
    {
        EXPECT_TRUE(
            encloses(css.global_bzone.exterior, css.global_bzone.interior));
    }

    // Intersect the given surfaces
    NodeId node_id
        = unit_builder_.insert_csg(Joined{op_and, std::move(css.nodes)}).first;

    TestResult result;
    result.node = build_infix_string(unit_.tree, node_id);
    result.surfaces = surface_strings(unit_);
    result.node_id = node_id;

    // Combine the bounding zones
    auto merged_bzone = calc_merged_bzone(css);
    result.interior = merged_bzone.interior;
    result.exterior = merged_bzone.exterior;

    return result;
}

//---------------------------------------------------------------------------//
void IntersectRegionTest::TestResult::print_expected() const
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
using BoxTest = IntersectRegionTest;

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

    EXPECT_EQ(SignedSense::inside,
              this->calc_sense(result.node_id, Real3{0, 0, 0}));
    EXPECT_EQ(SignedSense::on,
              this->calc_sense(result.node_id, Real3{1, 0, 0}));
    EXPECT_EQ(SignedSense::outside,
              this->calc_sense(result.node_id, Real3{0, 3, 0}));
    EXPECT_EQ(SignedSense::outside,
              this->calc_sense(result.node_id, Real3{0, 0, -4}));
}

//---------------------------------------------------------------------------//
// CONE
//---------------------------------------------------------------------------//
using ConeTest = IntersectRegionTest;

TEST_F(ConeTest, errors)
{
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

TEST_F(ConeTest, cylinder)
{
    auto result = this->test(Cone({1.2, 1.2}, 1.3 / 2));

    static char const expected_node[] = "all(+0, -1, -2)";
    static char const* const expected_surfaces[]
        = {"Plane: z=-0.65", "Plane: z=0.65", "Cyl z: r=1.2"};

    EXPECT_EQ(expected_node, result.node);
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    EXPECT_VEC_SOFT_EQ((Real3{-0.84852813742386, -0.84852813742386, -0.65}),
                       result.interior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{0.84852813742386, 0.84852813742386, 0.65}),
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
using CylinderTest = IntersectRegionTest;

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
using EllipsoidTest = IntersectRegionTest;

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
// GENTRAP
//---------------------------------------------------------------------------//
using GenTrapTest = IntersectRegionTest;

TEST_F(GenTrapTest, construct)
{
    // Validate contruction parameters
    EXPECT_THROW(GenTrap(-3,
                         {{-1, -1}, {-1, 1}, {1, 1}, {1, -1}},
                         {{-2, -2}, {-2, 2}, {2, 2}, {2, -2}}),
                 RuntimeError);  // negative dZ
    EXPECT_THROW(GenTrap(3,
                         {{-1, -1}, {-1, 1}, {1, 1}, {2, 0}, {1, -1}},
                         {{-2, -2}, {-2, 2}, {2, 2}, {2, -2}}),
                 RuntimeError);  // 5 pts in -dZ
    EXPECT_THROW(GenTrap(3,
                         {{-1, -1}, {0.4, -0.4}, {1, 1}, {1, -1}},
                         {{-2, -2}, {-2, 2}, {2, 2}, {2, -2}}),
                 RuntimeError);  // non-convex

    // Validate TRD-like construction parameters - 5 half-lengths
    EXPECT_THROW(GenTrap::from_trd(-3, {1, 1}, {2, 2}), RuntimeError);  // dZ<0
    EXPECT_THROW(GenTrap::from_trd(3, {-1, 1}, {2, 2}), RuntimeError);  // hx1<0
    EXPECT_THROW(GenTrap::from_trd(3, {1, -1}, {2, 2}), RuntimeError);  // hy1<0
    EXPECT_THROW(GenTrap::from_trd(3, {1, 1}, {-2, 2}), RuntimeError);  // hx2<0
    EXPECT_THROW(GenTrap::from_trd(3, {1, 1}, {2, -2}), RuntimeError);  // hy2<0
}

TEST_F(GenTrapTest, box_like)
{
    auto result = this->test(GenTrap(3,
                                     {{-1, -2}, {1, -2}, {1, 2}, {-1, 2}},
                                     {{-1, -2}, {1, -2}, {1, 2}, {-1, 2}}));

    static char const expected_node[] = "all(+0, -1, +2, -3, -4, +5)";
    static char const* const expected_surfaces[] = {"Plane: z=-3",
                                                    "Plane: z=3",
                                                    "Plane: y=-2",
                                                    "Plane: x=1",
                                                    "Plane: y=2",
                                                    "Plane: x=-1"};

    EXPECT_EQ(expected_node, result.node);
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    EXPECT_VEC_SOFT_EQ((Real3{-1, -2, -3}), result.interior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{1, 2, 3}), result.interior.upper());
    EXPECT_VEC_SOFT_EQ((Real3{-1, -2, -3}), result.exterior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{1, 2, 3}), result.exterior.upper());
}

TEST_F(GenTrapTest, trd1)
{
    auto result = this->test(GenTrap(3,
                                     {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}},
                                     {{-2, -2}, {2, -2}, {2, 2}, {-2, 2}}));

    static char const expected_node[] = "all(+0, -1, +2, -3, -4, +5)";
    static char const* const expected_surfaces[]
        = {"Plane: z=-3",
           "Plane: z=3",
           "Plane: n={0,0.98639,0.1644}, d=-1.4796",
           "Plane: n={0.98639,0,-0.1644}, d=1.4796",
           "Plane: n={0,0.98639,-0.1644}, d=1.4796",
           "Plane: n={0.98639,0,0.1644}, d=-1.4796"};

    EXPECT_EQ(expected_node, result.node);
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    EXPECT_FALSE(result.interior) << result.interior;
    EXPECT_VEC_SOFT_EQ((Real3{-2, -2, -3}), result.exterior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{2, 2, 3}), result.exterior.upper());
}

TEST_F(GenTrapTest, trd2)
{
    auto result = this->test(GenTrap::from_trd(3, {1, 1}, {2, 2}));

    static char const expected_node[] = "all(+0, -1, +2, -3, -4, +5)";
    static char const* const expected_surfaces[]
        = {"Plane: z=-3",
           "Plane: z=3",
           "Plane: n={0,0.98639,0.1644}, d=-1.4796",
           "Plane: n={0.98639,0,-0.1644}, d=1.4796",
           "Plane: n={0,0.98639,-0.1644}, d=1.4796",
           "Plane: n={0.98639,0,0.1644}, d=-1.4796"};

    EXPECT_EQ(expected_node, result.node);
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    EXPECT_FALSE(result.interior) << result.interior;
    EXPECT_VEC_SOFT_EQ((Real3{-2, -2, -3}), result.exterior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{2, 2, 3}), result.exterior.upper());
}

TEST_F(GenTrapTest, ppiped)
{
    auto result = this->test(GenTrap(4,
                                     {{-2, -2}, {0, -2}, {0, 0}, {-2, 0}},
                                     {{0, 0}, {2, 0}, {2, 2}, {0, 2}}));

    static char const expected_node[] = "all(+0, -1, +2, -3, -4, +5)";
    static char const* const expected_surfaces[]
        = {"Plane: z=-4",
           "Plane: z=4",
           "Plane: n={0,0.97014,-0.24254}, d=-0.97014",
           "Plane: n={0.97014,0,-0.24254}, d=0.97014",
           "Plane: n={0,0.97014,-0.24254}, d=0.97014",
           "Plane: n={0.97014,0,-0.24254}, d=-0.97014"};

    EXPECT_EQ(expected_node, result.node);
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    EXPECT_FALSE(result.interior) << result.interior;
    EXPECT_VEC_SOFT_EQ((Real3{-2, -2, -4}), result.exterior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{2, 2, 4}), result.exterior.upper());
}

TEST_F(GenTrapTest, triang_prism)
{
    auto result = this->test(
        GenTrap(3, {{-1, -1}, {-1, 1}, {2, 0}}, {{-1, -1}, {-1, 1}, {2, 0}}));

    static char const expected_node[] = "all(+0, -1, -2, +3, -4)";
    static char const* const expected_surfaces[] = {
        "Plane: z=-3",
        "Plane: z=3",
        "Plane: n={0.31623,0.94868,-0}, d=0.63246",
        "Plane: x=-1",
        "Plane: n={0.31623,-0.94868,0}, d=0.63246",
    };

    EXPECT_EQ(expected_node, result.node);
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    EXPECT_FALSE(result.interior) << result.interior;
    EXPECT_VEC_SOFT_EQ((Real3{-1, -1, -3}), result.exterior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{2, 1, 3}), result.exterior.upper());
}

TEST_F(GenTrapTest, trap_corners)
{
    auto result
        = this->test(GenTrap(40,
                             {{-19, -30}, {-19, 30}, {21, 30}, {21, -30}},
                             {{-21, -30}, {-21, 30}, {19, 30}, {19, -30}}));

    static char const expected_node[] = "all(+0, -1, -2, -3, +4, +5)";
    static char const* const expected_surfaces[] = {
        "Plane: z=-40",
        "Plane: z=40",
        "Plane: n={0.99969,-0,0.024992}, d=19.994",
        "Plane: y=30",
        "Plane: n={0.99969,0,0.024992}, d=-19.994",
        "Plane: y=-30",
    };

    EXPECT_EQ(expected_node, result.node);
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    EXPECT_FALSE(result.interior) << result.interior;
    EXPECT_VEC_SOFT_EQ((Real3{-21, -30, -40}), result.exterior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{21, 30, 40}), result.exterior.upper());
}

TEST_F(GenTrapTest, trapezoid_trans)
{
    // trapezoid but translated -30, -30
    auto result
        = this->test(GenTrap(40,
                             {{-49, -60}, {-49, 0}, {-9, 0}, {-9, -60}},
                             {{-51, -60}, {-51, 0}, {-11, 0}, {-11, -60}}));

    static char const expected_node[] = "all(+0, -1, -2, -3, +4, +5)";
    static char const* const expected_surfaces[] = {
        "Plane: z=-40",
        "Plane: z=40",
        "Plane: n={0.99969,-0,0.024992}, d=-9.9969",
        "Plane: y=0",
        "Plane: n={0.99969,0,0.024992}, d=-49.984",
        "Plane: y=-60",
    };

    EXPECT_EQ(expected_node, result.node);
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    EXPECT_FALSE(result.interior) << result.interior;
    EXPECT_VEC_SOFT_EQ((Real3{-51, -60, -40}), result.exterior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{-9, 0, 40}), result.exterior.upper());
}

TEST_F(GenTrapTest, trapezoid_ccw)
{
    auto result
        = this->test(GenTrap(40,
                             {{-19, -30}, {21, -30}, {21, 30}, {-19, 30}},
                             {{-21, -30}, {19, -30}, {19, 30}, {-21, 30}}));

    static char const expected_node[] = "all(+0, -1, +2, -3, -4, +5)";
    static char const* const expected_surfaces[]
        = {"Plane: z=-40",
           "Plane: z=40",
           "Plane: y=-30",
           "Plane: n={0.99969,-0,0.024992}, d=19.994",
           "Plane: y=30",
           "Plane: n={0.99969,0,0.024992}, d=-19.994"};

    EXPECT_EQ(expected_node, result.node);
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    EXPECT_FALSE(result.interior) << result.interior;
    EXPECT_VEC_SOFT_EQ((Real3{-21, -30, -40}), result.exterior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{21, 30, 40}), result.exterior.upper());
}

TEST_F(GenTrapTest, trap_theta)
{
    auto result = this->test(GenTrap::from_trap(
        40, Turn{0.125}, Turn{0}, {20, 10, 10, 0}, {20, 10, 10, 0}));

    static char const expected_node[] = "all(+0, -1, +2, -3, -4, +5)";
    static char const* const expected_surfaces[]
        = {"Plane: z=-40",
           "Plane: z=40",
           "Plane: y=-20",
           "Plane: n={0.70711,0,-0.70711}, d=7.0711",
           "Plane: y=20",
           "Plane: n={0.70711,0,-0.70711}, d=-7.0711"};

    EXPECT_EQ(expected_node, result.node);
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    EXPECT_FALSE(result.interior) << result.interior;
    EXPECT_VEC_SOFT_EQ((Real3{-50, -20, -40}), result.exterior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{50, 20, 40}), result.exterior.upper());
}

TEST_F(GenTrapTest, trap_thetaphi)
{
    auto result = this->test(GenTrap::from_trap(
        40, Turn{0.125}, Turn{0.25}, {20, 10, 10, 0}, {20, 10, 10, 0}));

    static char const expected_node[] = "all(+0, -1, +2, -3, -4, +5)";
    static char const* const expected_surfaces[]
        = {"Plane: z=-40",
           "Plane: z=40",
           "Plane: n={0,0.70711,-0.70711}, d=-14.142",
           "Plane: x=10",
           "Plane: n={0,0.70711,-0.70711}, d=14.142",
           "Plane: x=-10"};

    EXPECT_EQ(expected_node, result.node);
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    EXPECT_FALSE(result.interior) << result.interior;
    EXPECT_VEC_SOFT_EQ((Real3{-10, -60, -40}), result.exterior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{10, 60, 40}), result.exterior.upper());
}

TEST_F(GenTrapTest, trap_g4)
{
    constexpr Turn degree{real_type{1} / 360};
    real_type tan_alpha = std::tan(15 * native_value_from(degree));

    auto result = this->test(GenTrap::from_trap(4,
                                                5 * degree,
                                                10 * degree,
                                                {2, 1, 1, tan_alpha},
                                                {3, 1.5, 1.5, tan_alpha}));
    static char const expected_node[] = "all(+0, -1, +2, -3, -4, +5)";
    static char const* const expected_surfaces[]
        = {"Plane: z=-4",
           "Plane: z=4",
           "Plane: n={0,0.99403,0.10915}, d=-2.4851",
           "Plane: n={0.95664,-0.25633,-0.13832}, d=1.1958",
           "Plane: n={0,0.99032,-0.13883}, d=2.4758",
           "Plane: n={0.96575,-0.25877,-0.018918}, d=-1.2072"};

    EXPECT_EQ(expected_node, result.node);
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    EXPECT_FALSE(result.interior) << result.interior;
    EXPECT_VEC_SOFT_EQ((Real3{-1.95920952072934, -2.93923101204883, -4}),
                       result.exterior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{2.64848563385739, 3.06076898795117, 4}),
                       result.exterior.upper());
}

TEST_F(GenTrapTest, trap_full)
{
    auto result = this->test(GenTrap::from_trap(
        40, Turn{0.125}, Turn{0.125}, {20, 10, 10, 0.1}, {20, 10, 10, 0.1}));

    static char const expected_node[] = "all(+0, -1, +2, -3, -4, +5)";
    static char const* const expected_surfaces[]
        = {"Plane: z=-40",
           "Plane: z=40",
           "Plane: n={0,0.8165,-0.57735}, d=-16.33",
           "Plane: n={0.84066,-0.084066,-0.53499}, d=8.4066",
           "Plane: n={0,0.8165,-0.57735}, d=16.33",
           "Plane: n={0.84066,-0.084066,-0.53499}, d=-8.4066"};

    EXPECT_EQ(expected_node, result.node);
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    EXPECT_FALSE(result.interior) << result.interior;
    EXPECT_VEC_SOFT_EQ((Real3{-40.2842712474619, -48.2842712474619, -40}),
                       result.exterior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{40.2842712474619, 48.2842712474619, 40}),
                       result.exterior.upper());
}

// TODO: this should be valid
TEST_F(GenTrapTest, DISABLED_pentahedron)
{
    auto result = this->test(
        GenTrap(3, {{-2, -2}, {3, 0}, {-2, 2}}, {{-2, -1}, {-1, 1}, {2, 0}}));
    result.print_expected();
}

// TODO: we may need to support this
TEST_F(GenTrapTest, DISABLED_tetrahedron)
{
    auto result = this->test(
        GenTrap(3, {{-1, -1}, {2, 0}, {-1, 1}}, {{0, 0}, {0, 0}, {0, 0}}));
}

// TODO: find a valid set of points
TEST_F(GenTrapTest, full)
{
    auto result = this->test(GenTrap(4,
                                     {{-2, -2}, {-2, 2}, {2, 2}, {2, -2}},
                                     {{-2, -2}, {-1, 1}, {1, 1}, {2, -2}}));

    static char const expected_node[] = "all(+0, -1, -2, -3, -4, +5)";
    static char const* const expected_surfaces[]
        = {"Plane: z=-4",
           "Plane: z=4",
           "GQuadric: {0,0,0} {-0.125,0.125,0} {3.5,0.5,0.5} -6",
           "Plane: n={0,0.99228,0.12403}, d=1.4884",
           "GQuadric: {0,0,0} {0.125,0.125,0} {-3.5,0.5,0.5} -6",
           "Plane: y=-2"};

    EXPECT_EQ(expected_node, result.node);
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    EXPECT_FALSE(result.interior) << result.interior;
    EXPECT_VEC_SOFT_EQ((Real3{-2, -2, -4}), result.exterior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{2, 2, 4}), result.exterior.upper());
}

TEST_F(GenTrapTest, full2)
{
    auto result = this->test(GenTrap::from_trap(
        40, Turn{0.125}, Turn{0}, {20, 10, 10, 0.1}, {20, 10, 15, -0.2}));

    static char const expected_node[] = "all(+0, -1, +2, -3, -4, +5)";
    static char const* const expected_surfaces[]
        = {"Plane: z=-40",
           "Plane: z=40",
           "Plane: y=-20",
           "GQuadric: {0,0,0} {0,0.0875,0} {40,-0.5,-41.25} -450",
           "Plane: y=20",
           "GQuadric: {0,0,0} {0,0.2125,0} {40,4.5,-38.75} 450"};

    EXPECT_EQ(expected_node, result.node);
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    EXPECT_FALSE(result.interior) << result.interior;
    EXPECT_VEC_SOFT_EQ((Real3{-52, -20, -40}), result.exterior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{54, 20, 40}), result.exterior.upper());
}

/*!
 * Test deduplication of two opposing quadric surfaces.
 *
 * \verbatim
 * Lower polygons:      Upper polygons:
 *
 * x=-1      x=1           x=-0.5
 * +----+----+ y=1      +--+------+ y=1
 * |    |    |          |   \     |
 * |    |  R |          |    \  R |
 * |  L |    |          |  L  \   |
 * |    |    |          |      \  |
 * +----+----+ y=-1     +-------+-+ y=-1
 *      x=0                     x=0.5
 * \endverbatim
 */
TEST_F(GenTrapTest, adjacent_twisted)
{
    {
        // Left
        auto result
            = this->test(GenTrap(1,
                                 {{-1, -1}, {0, -1}, {0, 1}, {-1, 1}},
                                 {{-1, -1}, {0.5, -1}, {-0.5, 1}, {-1, 1}}));

        static char const expected_node[] = "all(+0, -1, +2, -3, -4, +5)";

        EXPECT_EQ(expected_node, result.node);
        EXPECT_VEC_SOFT_EQ((Real3{-1, -1, -1}), result.exterior.lower());
        EXPECT_VEC_SOFT_EQ((Real3{0.5, 1, 1}), result.exterior.upper());
    }
    {
        // Right
        auto result
            = this->test(GenTrap(1,
                                 {{0, -1}, {1, -1}, {1, 1}, {0, 1}},
                                 {{0.5, -1}, {1, -1}, {1, 1}, {-0.5, 1}}));

        static char const expected_node[] = "all(+0, -1, +2, +3, -4, -6)";

        EXPECT_EQ(expected_node, result.node);
        EXPECT_VEC_SOFT_EQ((Real3{-0.5, -1, -1}), result.exterior.lower());
        EXPECT_VEC_SOFT_EQ((Real3{1, 1, 1}), result.exterior.upper());
    }
    {
        // Scaled (broadened) right side with the same hyperboloid but
        // different size
        // TODO: the scaled GQ should be normalized
        auto result = this->test(GenTrap(1,
                                         {{0, -2}, {2, -2}, {2, 2}, {0, 2}},
                                         {{1, -2}, {2, -2}, {2, 2}, {-1, 2}}));
        static char const expected_node[] = "all(+0, -1, +7, -8, -9, +10)";

        EXPECT_EQ(expected_node, result.node);
        EXPECT_VEC_SOFT_EQ((Real3{-1, -2, -1}), result.exterior.lower());
        EXPECT_VEC_SOFT_EQ((Real3{2, 2, 1}), result.exterior.upper());
    }

    static char const* const expected_surfaces[] = {
        "Plane: z=-1",
        "Plane: z=1",
        "Plane: y=-1",
        "GQuadric: {0,0,0} {0,0.5,0} {2,0.5,0} 0",
        "Plane: y=1",
        "Plane: x=-1",
        "Plane: x=1",
        "Plane: y=-2",
        "Plane: x=2",
        "Plane: y=2",
        "GQuadric: {0,0,0} {0,1,0} {4,1,0} 0",
    };
    EXPECT_VEC_EQ(expected_surfaces, surface_strings(this->unit()));
}

//---------------------------------------------------------------------------//
// INFWEDGE
//---------------------------------------------------------------------------//
using InfWedgeTest = IntersectRegionTest;

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
// PARALLELEPIPED
//---------------------------------------------------------------------------//
using ParallelepipedTest = IntersectRegionTest;

TEST_F(ParallelepipedTest, errors)
{
    EXPECT_THROW(Parallelepiped({0, 1, 2}, Turn(0.1), Turn(0.1), Turn(0.1)),
                 RuntimeError);  // bad x
    EXPECT_THROW(Parallelepiped({2, 0, 1}, Turn(0.2), Turn(0.0), Turn(0.1)),
                 RuntimeError);  // bad y
    EXPECT_THROW(Parallelepiped({2, 1, 0}, Turn(0.1), Turn(0.1), Turn(0.1)),
                 RuntimeError);  // bad z

    Real3 sides{1, 2, 3};
    EXPECT_THROW(Parallelepiped(sides, Turn(0.3), Turn(0.1), Turn(0.1)),
                 RuntimeError);  // alpha
    EXPECT_THROW(Parallelepiped(sides, Turn(0.1), Turn(0.3), Turn(0.1)),
                 RuntimeError);  // theta
    EXPECT_THROW(Parallelepiped(sides, Turn(0.1), Turn(0.1), Turn(1.0)),
                 RuntimeError);  // phi
}

TEST_F(ParallelepipedTest, box)
{
    Real3 sides{1, 2, 3};
    auto result
        = this->test(Parallelepiped(sides, Turn(0.0), Turn(0.0), Turn(0.0)));

    static char const expected_node[] = "all(+0, -1, +2, -3, +4, -5)";
    static char const* const expected_surfaces[] = {"Plane: z=-3",
                                                    "Plane: z=3",
                                                    "Plane: y=-2",
                                                    "Plane: y=2",
                                                    "Plane: x=-1",
                                                    "Plane: x=1"};

    EXPECT_EQ(expected_node, result.node);
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    EXPECT_VEC_SOFT_EQ((Real3{-1, -2, -3}), result.interior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{1, 2, 3}), result.interior.upper());
    EXPECT_VEC_SOFT_EQ((Real3{-1, -2, -3}), result.exterior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{1, 2, 3}), result.exterior.upper());
}

TEST_F(ParallelepipedTest, alpha)
{
    Real3 sides{1, 2, 3};
    auto result
        = this->test(Parallelepiped(sides, Turn(0.1), Turn(0.0), Turn(0.0)));

    static char const expected_node[] = "all(+0, -1, +2, -3, +4, -5)";
    static char const* const expected_surfaces[]
        = {"Plane: z=-3",
           "Plane: z=3",
           "Plane: y=-1.618",
           "Plane: y=1.618",
           "Plane: n={0.80902,-0.58779,0}, d=-0.80902",
           "Plane: n={0.80902,-0.58779,0}, d=0.80902"};

    EXPECT_EQ(expected_node, result.node);
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    EXPECT_FALSE(result.interior) << result.interior;
    EXPECT_VEC_SOFT_EQ((Real3{-2.1755705045849, -1.6180339887499, -3}),
                       result.exterior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{2.1755705045849, 1.6180339887499, 3}),
                       result.exterior.upper());
}

TEST_F(ParallelepipedTest, theta)
{
    Real3 sides{1, 2, 3};
    auto result
        = this->test(Parallelepiped(sides, Turn(0), Turn(0.1), Turn(0)));

    static char const expected_node[] = "all(+0, -1, +2, -3, +4, -5)";
    static char const* const expected_surfaces[]
        = {"Plane: z=-3",
           "Plane: z=3",
           "Plane: y=-2",
           "Plane: y=2",
           "Plane: n={0.80902,0,-0.58779}, d=-0.80902",
           "Plane: n={0.80902,0,-0.58779}, d=0.80902"};

    EXPECT_EQ(expected_node, result.node);
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    EXPECT_FALSE(result.interior) << result.interior;
    EXPECT_VEC_SOFT_EQ((Real3{-2.7633557568774, -2, -2.4270509831248}),
                       result.exterior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{2.7633557568774, 2, 2.4270509831248}),
                       result.exterior.upper());
}

TEST_F(ParallelepipedTest, full)
{
    Real3 sides{1, 2, 3};
    auto result
        = this->test(Parallelepiped(sides, Turn(0.1), Turn(0.05), Turn(0.15)));

    static char const expected_node[] = "all(+0, -1, +2, -3, +4, -5)";
    static char const* const expected_surfaces[]
        = {"Plane: z=-3",
           "Plane: z=3",
           "Plane: n={0,0.96714,-0.25423}, d=-1.5649",
           "Plane: n={0,0.96714,-0.25423}, d=1.5649",
           "Plane: n={0.80902,-0.58779,0}, d=-0.80902",
           "Plane: n={0.80902,-0.58779,0}, d=0.80902"};

    EXPECT_EQ(expected_node, result.node);
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    EXPECT_FALSE(result.interior) << result.interior;
    EXPECT_VEC_SOFT_EQ(
        (Real3{-2.720477400589, -2.3680339887499, -2.8531695488855}),
        result.exterior.lower());
    EXPECT_VEC_SOFT_EQ(
        (Real3{2.720477400589, 2.3680339887499, 2.8531695488855}),
        result.exterior.upper());
}

//---------------------------------------------------------------------------//
// PRISM
//---------------------------------------------------------------------------//
using PrismTest = IntersectRegionTest;

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
using SphereTest = IntersectRegionTest;

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
