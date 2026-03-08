//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/IntersectRegion.test.cc
//---------------------------------------------------------------------------//
#include "orange/orangeinp/IntersectRegion.hh"

#include "corecel/io/Join.hh"
#include "corecel/io/Logger.hh"
#include "orange/BoundingBoxUtils.hh"
#include "orange/MatrixUtils.hh"
#include "orange/OrangeTypes.hh"
#include "orange/orangeinp/CsgTreeUtils.hh"
#include "orange/orangeinp/IntersectSurfaceBuilder.hh"
#include "orange/orangeinp/detail/CsgUnitBuilder.hh"
#include "orange/orangeinp/detail/IntersectSurfaceState.hh"
#include "orange/orangeinp/detail/SenseEvaluator.hh"

#include "CsgTestUtils.hh"
#include "IntersectTestResult.hh"
#include "celeritas_test.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
std::ostream& operator<<(std::ostream& os, SignedSense s)
{
    return (os << to_cstring(s));
}

Turn atan_to_turn(real_type v)
{
    return native_value_to<Turn>(std::atan(v));
}

namespace orangeinp
{
namespace test
{
//---------------------------------------------------------------------------//
class IntersectRegionTest : public ::celeritas::test::Test
{
  public:
    using Unit = orangeinp::detail::CsgUnit;
    using UnitBuilder = orangeinp::detail::CsgUnitBuilder;
    using State = orangeinp::detail::IntersectSurfaceState;
    using Tol = UnitBuilder::Tol;
    using TestResult = IntersectTestResult;

  protected:
    // Build with an explicit name and transform
    State build(std::string name,
                IntersectRegionInterface const& r,
                VariantTransform const& vt);

    // Insert a built state as a new "volume"
    NodeId insert(State&& css);

    // Test with an explicit name and transform
    TestResult test(std::string&& name,
                    IntersectRegionInterface const& r,
                    VariantTransform const& vt);

    //! Test with default name
    TestResult
    test(IntersectRegionInterface const& r, VariantTransform const& vt)
    {
        return this->test("cr", r, vt);
    }

    //! Test with no transform
    TestResult test(IntersectRegionInterface const& r)
    {
        return this->test(r, NoTransformation{});
    }

    //! Test with no transform
    TestResult test(std::string&& name, IntersectRegionInterface const& r)
    {
        return this->test(std::move(name), r, NoTransformation{});
    }

    SignedSense calc_sense(NodeId n, Real3 const& pos) const
    {
        CELER_EXPECT(n < unit_.tree.size());
        LocalSurfaceId on_surface;
        detail::SenseEvaluator eval_sense(
            unit_.tree, unit_.surfaces, pos, &on_surface);
        auto result = eval_sense(n);
        if (on_surface)
        {
            CELER_EXPECT(n < unit_.metadata.size());
            auto const& md_set = unit_.metadata[n.get()];
            CELER_LOG(debug)
                << "Point " << repr(pos) << " is on surface "
                << on_surface.get() << " of node " << n.get() << " = "
                << join(md_set.begin(), md_set.end(), ',');
        }
        return result;
    }

    Unit const& unit() const { return unit_; }
    Tol const& tol() const { return unit_builder_.tol(); }

    void reset_with_tol(Tol const& t)
    {
        CELER_EXPECT(t);
        unit_ = {};
        unit_builder_ = UnitBuilder{&unit_, t, BBox::from_infinite()};
    }

  private:
    Unit unit_;
    UnitBuilder unit_builder_{
        &unit_, Tol::from_relative(1e-4), BBox::from_infinite()};
};

//---------------------------------------------------------------------------//
auto IntersectRegionTest::build(std::string name,
                                IntersectRegionInterface const& r,
                                VariantTransform const& trans) -> State
{
    State css;
    css.transform = &trans;
    css.make_face_name = {};
    css.object_name = std::move(name);

    IntersectSurfaceBuilder insert_surface{&unit_builder_, &css};
    r.build(insert_surface);
    return css;
}

//---------------------------------------------------------------------------//
NodeId IntersectRegionTest::insert(State&& css)
{
    auto node_id
        = unit_builder_.insert_csg(Joined{op_and, std::move(css.nodes)}).first;
    unit_builder_.insert_md(node_id, std::move(css.object_name));
    unit_.tree.insert_volume(node_id);
    std::move(css) = {};
    return node_id;
}

//---------------------------------------------------------------------------//
auto IntersectRegionTest::test(std::string&& name,
                               IntersectRegionInterface const& r,
                               VariantTransform const& trans) -> TestResult
{
    // Intersect the given surfaces
    auto css = this->build(std::move(name), r, trans);
    // Save bounding zone
    auto merged_bzone = calc_merged_bzone(css);
    // Build CSG node+md
    auto node_id = this->insert(std::move(css));

    TestResult result;
    result.node = build_infix_string(unit_.tree, node_id);
    result.surfaces = surface_strings(unit_);
    result.interior = merged_bzone.interior;
    result.exterior = merged_bzone.exterior;
    result.node_id = node_id;

    return result;
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
    static char const* const expected_surfaces[] = {
        "Plane: x=-1",
        "Plane: x=1",
        "Plane: y=-2",
        "Plane: y=2",
        "Plane: z=-3",
        "Plane: z=3",
    };

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
    EXPECT_THROW(Cone({0, 0}, 1), RuntimeError);
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
        = {"Plane: z=-10", "Plane: z=10", "Cone z: t=5e-3 at {0,0,100}"};

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
// CUTCYLINDER
//---------------------------------------------------------------------------//
using CutCylinderTest = IntersectRegionTest;

TEST_F(CutCylinderTest, errors)
{
    real_type k = std::sqrt(2) / 2;

    EXPECT_THROW(CutCylinder(0.0, 1.0, {k, 0, -k}, {k, 0, k}), RuntimeError);
    EXPECT_THROW(CutCylinder(1.0, -1.0, {k, 0, -k}, {k, 0, k}), RuntimeError);
    EXPECT_THROW(CutCylinder(1.0, 1.0, {k, 0, k}, {0, 0, k}), RuntimeError);
    EXPECT_THROW(CutCylinder(1.0, 1.0, {0, 0, -k}, {0, 0, -k}), RuntimeError);
    EXPECT_THROW(CutCylinder(1.0, 1.0, {0, 0.5, -0.5}, {0, k, -k}),
                 RuntimeError);
}

TEST_F(CutCylinderTest, encloses)
{
    real_type k = std::sqrt(2) / 2;
    CutCylinder cyl1(1.0, 1.0, {k, 0, -k}, {k, 0, k});

    EXPECT_TRUE(cyl1.encloses(CutCylinder(0.9, 0.9, {k, 0, -k}, {k, 0, k})));
    EXPECT_FALSE(cyl1.encloses(CutCylinder(0.9, 1.9, {k, 0, -k}, {k, 0, k})));
    EXPECT_FALSE(cyl1.encloses(CutCylinder(1.9, 0.9, {k, 0, -k}, {k, 0, k})));

    EXPECT_THROW(cyl1.encloses(CutCylinder(0.9, 0.9, {k, 0, -k}, {0, k, k})),
                 RuntimeError);
    EXPECT_THROW(cyl1.encloses(CutCylinder(0.9, 0.9, {0, k, -k}, {k, 0, k})),
                 RuntimeError);
}

TEST_F(CutCylinderTest, standard)
{
    real_type k = std::sqrt(2) / 2;

    auto result = this->test(CutCylinder(0.75, 0.9, {0, k, -k}, {-k, 0, k}));

    static char const expected_node[] = "all(-0, +1, -2)";
    static char const* const expected_surfaces[]
        = {"Plane: n={0,0.70711,-0.70711}, d=0.63640",
           "Plane: n={0.70711,0,-0.70711}, d=-0.63640",
           "Cyl z: r=0.75"};

    EXPECT_EQ(expected_node, result.node);
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    EXPECT_VEC_SOFT_EQ((Real3{-0.75, -0.75, -0.9}), result.exterior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{0.75, 0.75, 0.9}), result.exterior.upper());
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

TEST_F(EllipsoidTest, encloses)
{
    Ellipsoid ellipsoid({1, 2, 3});
    EXPECT_TRUE(ellipsoid.encloses(Ellipsoid({1, 2, 3})));
    EXPECT_TRUE(ellipsoid.encloses(Ellipsoid({0.5, 1.5, 2.5})));
    EXPECT_FALSE(ellipsoid.encloses(Ellipsoid({0.5, 1.5, 3.5})));
    EXPECT_FALSE(ellipsoid.encloses(Ellipsoid({0.5, 2.5, 2.5})));
    EXPECT_FALSE(ellipsoid.encloses(Ellipsoid({5.5, 1.5, 2.5})));
}

TEST_F(EllipsoidTest, standard)
{
    auto result = this->test(Ellipsoid({3, 2, 1}));

    static char const expected_node[] = "-0";
    static char const* const expected_surfaces[]
        = {"SQuadric: {0.33333,0.75,3} {0,0,0} -3"};

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

TEST_F(EllipsoidTest, tiny)
{
    auto result = this->test(Ellipsoid({0.008, 0.004, 0.005}));

    static char const expected_node[] = "-0";
    static char const* const expected_surfaces[]
        = {"SQuadric: {0.5,2,1.28} {0,0,0} -3.2e-5"};

    EXPECT_EQ(expected_node, result.node);
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
}

//---------------------------------------------------------------------------//
// ELLIPTICAL CYLINDER
//---------------------------------------------------------------------------//
using EllipticalCylinderTest = IntersectRegionTest;

TEST_F(EllipticalCylinderTest, errors)
{
    EXPECT_THROW(EllipticalCylinder({1, -1}, 2), RuntimeError);
    EXPECT_THROW(EllipticalCylinder({1, 2}, -2), RuntimeError);
}

TEST_F(EllipticalCylinderTest, encloses)
{
    EllipticalCylinder ec({1, 2}, 3);
    EXPECT_TRUE(ec.encloses(EllipticalCylinder({1, 2}, 3)));
    EXPECT_FALSE(ec.encloses(EllipticalCylinder({1.1, 2.1}, 3.1)));
    EXPECT_FALSE(ec.encloses(EllipticalCylinder({1.1, 2}, 3)));
    EXPECT_FALSE(ec.encloses(EllipticalCylinder({1, 2.1}, 3)));
    EXPECT_FALSE(ec.encloses(EllipticalCylinder({1, 2}, 3.1)));
}

TEST_F(EllipticalCylinderTest, standard)
{
    auto result = this->test(EllipticalCylinder({3, 2}, 0.5));

    static char const expected_node[] = "all(+0, -1, -2)";
    static char const* const expected_surfaces[] = {
        "Plane: z=-0.5",
        "Plane: z=0.5",
        "SQuadric: {0.66667,1.5,0} {0,0,0} -6",
    };

    EXPECT_EQ(expected_node, result.node);
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    EXPECT_VEC_SOFT_EQ((Real3{-2.1213203435596424, -1.414213562373095, -0.5}),
                       result.interior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{2.1213203435596424, 1.414213562373095, 0.5}),
                       result.interior.upper());
    EXPECT_VEC_SOFT_EQ((Real3{-3, -2, -0.5}), result.exterior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{3, 2, 0.5}), result.exterior.upper());
}

//---------------------------------------------------------------------------//
// ELLIPTICAL CONE
//---------------------------------------------------------------------------//
using EllipticalConeTest = IntersectRegionTest;

TEST_F(EllipticalConeTest, errors)
{
    // Negatives
    EXPECT_THROW(EllipticalCone({-1, 5}, {1, 3}, 2), RuntimeError);
    EXPECT_THROW(EllipticalCone({1, -5}, {1, 3}, 2), RuntimeError);
    EXPECT_THROW(EllipticalCone({1, 3}, {-1, 5}, 2), RuntimeError);
    EXPECT_THROW(EllipticalCone({1, 3}, {1, -5}, 2), RuntimeError);
    EXPECT_THROW(EllipticalCone({1, 3}, {1, 5}, -2), RuntimeError);

    // Partial zeros
    EXPECT_THROW(EllipticalCone({0, 5}, {1, 3}, 2), RuntimeError);
    EXPECT_THROW(EllipticalCone({1, 0}, {1, 3}, 2), RuntimeError);
    EXPECT_THROW(EllipticalCone({3, 1}, {0, 3}, 2), RuntimeError);
    EXPECT_THROW(EllipticalCone({3, 1}, {1, 0}, 2), RuntimeError);

    // Mismatched aspect ratios
    EXPECT_THROW(EllipticalCone({1, 3}, {1, 5}, 2), RuntimeError);
    EXPECT_THROW(EllipticalCone({1, 3}, {5, 1}, 2), RuntimeError);

    // Elliptical cylinder
    EXPECT_THROW(EllipticalCone({1, 3}, {1, 3}, 2), RuntimeError);
}

TEST_F(EllipticalConeTest, encloses)
{
    EllipticalCone ec({1, 2}, {3, 6}, 5);
    EXPECT_TRUE(ec.encloses(EllipticalCone({1, 2}, {3, 6}, 5)));
    EXPECT_TRUE(ec.encloses(EllipticalCone({0.5, 1.5}, {1, 3}, 5)));
    EXPECT_FALSE(ec.encloses(EllipticalCone({1, 2}, {3.1, 6.2}, 5)));
    EXPECT_FALSE(ec.encloses(EllipticalCone({0.8, 2}, {3, 7.5}, 5)));
    EXPECT_FALSE(ec.encloses(EllipticalCone({1, 2}, {3, 6}, 5.1)));
}

TEST_F(EllipticalConeTest, standard)
{
    auto result = this->test(EllipticalCone({1, 3}, {2, 6}, 3));

    static char const expected_node[] = "all(+0, -1, -2)";
    static char const* const expected_surfaces[]
        = {"Plane: z=-3", "Plane: z=3", "SQuadric: {36,4,-1} {0,0,-18} -81"};

    EXPECT_EQ(expected_node, result.node);
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);

    EXPECT_VEC_SOFT_EQ((Real3{-2, -6, -3}), result.exterior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{2, 6, 3}), result.exterior.upper());
}

TEST_F(EllipticalConeTest, vertex)
{
    auto result = this->test(EllipticalCone({0, 0}, {2, 4}, 4));

    static char const expected_node[] = "all(+0, -1, -2)";
    static char const* const expected_surfaces[]
        = {"Plane: z=-4", "Plane: z=4", "SQuadric: {16,4,-1} {0,0,-8} -16"};

    EXPECT_EQ(expected_node, result.node);
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);

    EXPECT_VEC_SOFT_EQ((Real3{-2, -4, -4}), result.exterior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{2, 4, 4}), result.exterior.upper());
}

//---------------------------------------------------------------------------//
// EXTRUDEDPOLYGON
//---------------------------------------------------------------------------//
using ExtrudedPolygonTest = IntersectRegionTest;

TEST_F(ExtrudedPolygonTest, simple_cube)
{
    // Test a simple unit cube
    ExtrudedPolygon::VecReal2 polygon{
        Real2{1, 0}, Real2{1, 1}, Real2{0, 1}, Real2{0, 0}};

    ExtrudedPolygon::PolygonFace bot{Real3{0, 0, 0}, 1};
    ExtrudedPolygon::PolygonFace top{Real3{0, 0, 1}, 1};

    auto result = this->test(ExtrudedPolygon(polygon, bot, top));

    static char const expected_node[] = "all(+0, -1, -2, -3, +4, +5)";
    static char const* const expected_surfaces[] = {
        "Plane: z=0",
        "Plane: z=1",
        "Plane: x=1",
        "Plane: y=1",
        "Plane: x=0",
        "Plane: y=0",
    };

    EXPECT_EQ(expected_node, result.node);
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);

    EXPECT_VEC_SOFT_EQ((Real3{0, 0, 0}), result.exterior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{1, 1, 1}), result.exterior.upper());
}

TEST_F(ExtrudedPolygonTest, collinear)
{
    // Same test as simple_cube, but with collinear points
    ExtrudedPolygon::VecReal2 polygon{
        Real2{0.3, 0},
        Real2{0.7, 0},
        Real2{1, 0},
        Real2{1, 0.5},
        Real2{1, 1},
        Real2{0.5, 1},
        Real2{0, 1},
        Real2{0, 0.5},
        Real2{0, 0},
    };

    ExtrudedPolygon::PolygonFace bot{Real3{0, 0, 0}, 1};
    ExtrudedPolygon::PolygonFace top{Real3{0, 0, 1}, 1};

    auto result = this->test(ExtrudedPolygon(polygon, bot, top));

    static char const expected_node[] = "all(+0, -1, -2, -3, +4, +5)";
    static char const* const expected_surfaces[] = {
        "Plane: z=0",
        "Plane: z=1",
        "Plane: x=1",
        "Plane: y=1",
        "Plane: x=0",
        "Plane: y=0",
    };

    EXPECT_EQ(expected_node, result.node);
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);

    EXPECT_VEC_SOFT_EQ((Real3{0, 0, 0}), result.exterior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{1, 1, 1}), result.exterior.upper());
}

TEST_F(ExtrudedPolygonTest, flat_top_pyramid)
{
    ExtrudedPolygon::VecReal2 polygon{
        Real2{1, 0}, Real2{1, 1}, Real2{0, 1}, Real2{0, 0}};

    ExtrudedPolygon::PolygonFace bot{Real3{0, 0, 0}, 1};
    ExtrudedPolygon::PolygonFace top{Real3{0, 0, 0.5}, 0.5};

    auto result = this->test(ExtrudedPolygon(polygon, bot, top));

    static char const expected_node[] = "all(+0, -1, -2, -3, +4, +5)";
    // Planes have x- and y-slopes equal to +/- sqrt(2)/2, as expected
    static char const* const expected_surfaces[] = {
        "Plane: z=0",
        "Plane: z=0.5",
        "Plane: n={0.70711,-0,0.70711}, d=0.70711",
        "Plane: n={0,0.70711,0.70711}, d=0.70711",
        "Plane: x=0",
        "Plane: y=0",
    };

    EXPECT_EQ(expected_node, result.node);
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);

    EXPECT_VEC_SOFT_EQ((Real3{0, 0, 0}), result.exterior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{1, 1, 0.5}), result.exterior.upper());
}

TEST_F(ExtrudedPolygonTest, skewed)
{
    // Irregular hexagon with a single collinear point at (0, 0)
    ExtrudedPolygon::VecReal2 polygon{Real2{1, 0},
                                      Real2{2, 2},
                                      Real2{1, 4},
                                      Real2{-1, 3},
                                      Real2{-2, 1},
                                      Real2{-1, 0},
                                      Real2{0, 0}};

    ExtrudedPolygon::PolygonFace bot{Real3{4, 3, 10}, 0.7};
    ExtrudedPolygon::PolygonFace top{Real3{10, 11, 15}, 0.5};

    auto result = this->test(ExtrudedPolygon(polygon, bot, top));

    static char const expected_node[] = "all(+0, -1, +2, -3, +4, -5, +6, +7)";
    static char const* const expected_surfaces[] = {
        "Plane: z=10",
        "Plane: z=15",
        "Plane: n={-0.85138,0.42569,0.30650}, d=0.34055",
        "Plane: n={0.45718,0.22859,-0.85950}, d=-5.1204",
        "Plane: n={0.35448,-0.70895,0.60970}, d=3.6511",
        "Plane: n={-0.81650,0.40825,0.40825}, d=3.4701",
        "Plane: n={0.31520,0.31520,-0.89516}, d=-6.9658",
        "Plane: n={0,0.53000,-0.84800}, d=-6.8900",
    };

    EXPECT_EQ(expected_node, result.node);
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);

    EXPECT_VEC_SOFT_EQ((Real3{2.6, 3, 10}), result.exterior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{11, 13, 15}), result.exterior.upper());
}

//---------------------------------------------------------------------------//
// GENPRISM
//---------------------------------------------------------------------------//
class GenPrismTest : public IntersectRegionTest
{
  protected:
    using VecReal = std::vector<real_type>;

    // NOTE: this only works for trapezoids centered on the z axis (a
    // requirement for Geant4 but not for ORANGE)
    void check_corners(NodeId nid, GenPrism const& pri, real_type bump) const
    {
        CELER_EXPECT(bump > 0);

        // Account for the center of the prism not being at the origin
        Real3 center{0, 0, 0};
        auto factor = 0.5 / pri.num_sides();
        for (auto i : range(pri.num_sides()))
        {
            auto const& lo = pri.lower()[i];
            auto const& hi = pri.upper()[i];
            center += factor * Real3{lo[0], lo[1], -pri.halfheight()};
            center += factor * Real3{hi[0], hi[1], +pri.halfheight()};
        }

        real_type const z[] = {-pri.halfheight(), pri.halfheight()};

        for (auto i : range(2))
        {
            auto const& points = (i == 0 ? pri.lower() : pri.upper());
            for (Real2 const& p : points)
            {
                Real3 const corner{p[0], p[1], z[i]};
                auto outward = make_unit_vector(corner - center);

                EXPECT_EQ(SignedSense::inside,
                          this->calc_sense(nid, corner - bump * outward))
                    << "inward by " << bump << " from " << repr(corner);
                EXPECT_EQ(SignedSense::outside,
                          this->calc_sense(nid, corner + bump * outward))
                    << "outward by " << bump << " from " << repr(corner);
            }
        }
    }

    //! Calculate the twist angles in fractions of a turn
    VecReal get_twist_angles(GenPrism const& pri) const
    {
        VecReal result;
        for (auto i : range(pri.num_sides()))
        {
            // Due to floating point errors in unit vector normalization, the
            // cosine could be *slightly* above 1.
            auto twist_cosine = pri.calc_twist_cosine(i);
            EXPECT_GT(twist_cosine, 0);
            EXPECT_LT(twist_cosine, 1 + SoftEqual<>{}.abs());
            real_type twist_angle
                = std::acos(std::fmin(twist_cosine, real_type(1)));
            result.push_back(native_value_to<Turn>(twist_angle).value());
        }
        return result;
    }

    VecReal to_vec(std::vector<Real2> const& inp) const
    {
        VecReal result;
        result.reserve(inp.size() * 2);
        for (auto const& v : inp)
        {
            result.insert(result.end(), v.begin(), v.end());
        }
        return result;
    }
};

TEST_F(GenPrismTest, construct)
{
    using RtErr = RuntimeError;
    // Validate construction parameters
    EXPECT_THROW(GenPrism(-3,
                          {{-1, -1}, {-1, 1}, {1, 1}, {1, -1}},
                          {{-2, -2}, {-2, 2}, {2, 2}, {2, -2}}),
                 RuntimeError);  // negative dZ
    EXPECT_THROW(GenPrism(3,
                          {{-1, -1}, {-1, 1}, {1, 1}, {2, 0}, {1, -1}},
                          {{-2, -2}, {-2, 2}, {2, 2}, {2, -2}}),
                 RuntimeError);  // incompatible number of points
    EXPECT_THROW(GenPrism(3,
                          {{-1, -1}, {0.4, -0.4}, {1, 1}, {1, -1}},
                          {{-2, -2}, {-2, 2}, {2, 2}, {2, -2}}),
                 RuntimeError);  // non-convex
    EXPECT_THROW(GenPrism(3,
                          {{-1, -2}, {1, -2}, {1, 2}, {-1, 2}},
                          {{-1, 2}, {1, 2}, {1, -2}, {-1, -2}}),
                 RuntimeError);  // different orientations
    EXPECT_THROW(GenPrism(2,
                          {{-0.5, 0}, {0.5, 0}, {0.5, 0}, {-0.5, 0}},
                          {{-0.5, 0}, {0.5, 0}, {0.5, 0}, {-0.5, 0}}),
                 RuntimeError);  // collinear top and bottom

    // Validate TRD-like construction parameters - 5 half-lengths
    EXPECT_THROW(GenPrism::from_trd(-3, {1, 1}, {2, 2}), RtErr);  // dZ<0
    EXPECT_THROW(GenPrism::from_trd(3, {-1, 1}, {2, 2}), RtErr);  // hx1<0
    EXPECT_THROW(GenPrism::from_trd(3, {1, -1}, {2, 2}), RtErr);  // hy1<0
    EXPECT_THROW(GenPrism::from_trd(3, {1, 1}, {-2, 2}), RtErr);  // hx2<0
    EXPECT_THROW(GenPrism::from_trd(3, {1, 1}, {2, -2}), RtErr);  // hy2<0
    EXPECT_THROW(GenPrism::from_trd(3, {0, 1}, {0, 2}), RtErr);  // degen x
    EXPECT_THROW(GenPrism::from_trd(3, {0, 1}, {1, 0}), RtErr);  // degen

    // Trap angles are invalid (note that we do *not* have the restriction of
    // Geant4 that the turns be the same: this just ends up creating a GenPrism
    // (with twisted sides) instead of a Trap
    EXPECT_THROW(
        GenPrism::from_trap(
            2, Turn{0}, Turn{0}, {2, 4, 4, Turn{-.26}}, {2, 4, 4, Turn{0.}}),
        RuntimeError);
    EXPECT_THROW(
        GenPrism::from_trap(
            2, Turn{0}, Turn{0}, {2, 4, 4, Turn{.27}}, {2, 4, 4, Turn{0.}}),
        RuntimeError);
    EXPECT_THROW(
        GenPrism::from_trap(
            2, Turn{0}, Turn{0}, {2, 4, 4, Turn{0}}, {2, 4, 4, Turn{0.25}}),
        RuntimeError);

    // Twist angle cannot be greater than 90 degrees
    EXPECT_THROW(GenPrism(1.0,
                          {{1, -1}, {1, 1}, {-1, 1}, {-1, -1}},
                          {{1, 1}, {-1, 1}, {-1, -1}, {1, -1}}),
                 RuntimeError);
}

TEST_F(GenPrismTest, box_like)
{
    GenPrism pri(3,
                 {{-1, -2}, {1, -2}, {1, 2}, {-1, 2}},
                 {{-1, -2}, {1, -2}, {1, 2}, {-1, 2}});

    static real_type const expected_twist_angles[] = {0, 0, 0, 0};
    EXPECT_VEC_SOFT_EQ(expected_twist_angles, this->get_twist_angles(pri));

    auto result = this->test(pri);

    static char const expected_node[] = "all(+0, -1, +2, -3, -4, +5)";
    static char const* const expected_surfaces[] = {
        "Plane: z=-3",
        "Plane: z=3",
        "Plane: y=-2",
        "Plane: x=1",
        "Plane: y=2",
        "Plane: x=-1",
    };

    EXPECT_EQ(expected_node, result.node);
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    EXPECT_VEC_SOFT_EQ((Real3{-1, -2, -3}), result.interior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{1, 2, 3}), result.interior.upper());
    EXPECT_VEC_SOFT_EQ((Real3{-1, -2, -3}), result.exterior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{1, 2, 3}), result.exterior.upper());
    this->check_corners(result.node_id, pri, 0.1);
}

TEST_F(GenPrismTest, ppiped)
{
    auto pri = GenPrism(4,
                        {{-2, -2}, {0, -2}, {0, 0}, {-2, 0}},
                        {{0, 0}, {2, 0}, {2, 2}, {0, 2}});
    auto result = this->test(pri);

    static char const expected_node[] = "all(+0, -1, +2, -3, -4, +5)";
    static char const* const expected_surfaces[] = {
        "Plane: z=-4",
        "Plane: z=4",
        "Plane: n={0,0.97014,-0.24254}, d=-0.97014",
        "Plane: n={0.97014,0,-0.24254}, d=0.97014",
        "Plane: n={0,0.97014,-0.24254}, d=0.97014",
        "Plane: n={0.97014,0,-0.24254}, d=-0.97014",
    };

    EXPECT_EQ(expected_node, result.node);
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    EXPECT_FALSE(result.interior) << result.interior;
    EXPECT_VEC_SOFT_EQ((Real3{-2, -2, -4}), result.exterior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{2, 2, 4}), result.exterior.upper());

    this->check_corners(result.node_id, pri, 0.1);
}

TEST_F(GenPrismTest, trap_corners)
{
    auto pri = GenPrism(40,
                        {{-19, -30}, {-19, 30}, {21, 30}, {21, -30}},
                        {{-21, -30}, {-21, 30}, {19, 30}, {19, -30}});
    auto result = this->test(pri);

    static char const expected_node[] = "all(+0, -1, -2, -3, +4, +5)";
    static char const* const expected_surfaces[] = {
        "Plane: z=-40",
        "Plane: z=40",
        "Plane: n={0.99969,0,0.024992}, d=19.994",
        "Plane: y=30",
        "Plane: n={0.99969,0,0.024992}, d=-19.994",
        "Plane: y=-30",
    };

    EXPECT_EQ(expected_node, result.node);
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    EXPECT_FALSE(result.interior) << result.interior;
    EXPECT_VEC_SOFT_EQ((Real3{-21, -30, -40}), result.exterior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{21, 30, 40}), result.exterior.upper());

    this->check_corners(result.node_id, pri, 1.0);
}

TEST_F(GenPrismTest, trapezoid_trans)
{
    // trapezoid but translated -30, -30
    auto pri = GenPrism(40,
                        {{-49, -60}, {-49, 0}, {-9, 0}, {-9, -60}},
                        {{-51, -60}, {-51, 0}, {-11, 0}, {-11, -60}});
    auto result = this->test(pri);

    static char const expected_node[] = "all(+0, -1, -2, -3, +4, +5)";
    static char const* const expected_surfaces[] = {
        "Plane: z=-40",
        "Plane: z=40",
        "Plane: n={0.99969,0,0.024992}, d=-9.9969",
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

TEST_F(GenPrismTest, trapezoid_ccw)
{
    auto pri = GenPrism(40,
                        {{-19, -30}, {21, -30}, {21, 30}, {-19, 30}},
                        {{-21, -30}, {19, -30}, {19, 30}, {-21, 30}});
    auto result = this->test(pri);

    static char const expected_node[] = "all(+0, -1, +2, -3, -4, +5)";
    static char const* const expected_surfaces[] = {
        "Plane: z=-40",
        "Plane: z=40",
        "Plane: y=-30",
        "Plane: n={0.99969,0,0.024992}, d=19.994",
        "Plane: y=30",
        "Plane: n={0.99969,0,0.024992}, d=-19.994",
    };

    EXPECT_EQ(expected_node, result.node);
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    EXPECT_FALSE(result.interior) << result.interior;
    EXPECT_VEC_SOFT_EQ((Real3{-21, -30, -40}), result.exterior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{21, 30, 40}), result.exterior.upper());

    this->check_corners(result.node_id, pri, 1.0);
}

TEST_F(GenPrismTest, full)
{
    GenPrism pri(4,
                 {{-2, -2}, {-2, 2}, {2, 2}, {2, -2}},
                 {{-2, -2}, {-1, 1}, {1, 1}, {2, -2}});

    static real_type const expected_twist_angles[]
        = {0.051208191174783, 0, 0.051208191174783, 0};
    EXPECT_VEC_SOFT_EQ(expected_twist_angles, this->get_twist_angles(pri));

    auto result = this->test(pri);
    static char const expected_node[] = "all(+0, -1, -2, -3, -4, +5)";
    static char const* const expected_surfaces[] = {
        "Plane: z=-4",
        "Plane: z=4",
        R"(GQuadric: {0,0,0} {0,0.035007,-0.035007} {0.98020,0.14003,0.14003} -1.6803)",
        "Plane: n={0,0.99228,0.12404}, d=1.4884",
        R"(GQuadric: {0,0,0} {0,0.035007,0.035007} {-0.98020,0.14003,0.14003} -1.6803)",
        "Plane: y=-2",
    };

    EXPECT_EQ(expected_node, result.node);
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    EXPECT_FALSE(result.interior) << result.interior;
    EXPECT_VEC_SOFT_EQ((Real3{-2, -2, -4}), result.exterior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{2, 2, 4}), result.exterior.upper());

    this->check_corners(result.node_id, pri, 0.01);
}

TEST_F(GenPrismTest, triang_prism)
{
    auto pri = GenPrism(
        3, {{-1, -1}, {-1, 1}, {2, 0}}, {{-1, -1}, {-1, 1}, {2, 0}});
    auto result = this->test(pri);

    static char const expected_node[] = "all(+0, -1, -2, +3, -4)";
    static char const* const expected_surfaces[] = {
        "Plane: z=-3",
        "Plane: z=3",
        "Plane: n={0.31623,0.94868,0}, d=0.63246",
        "Plane: x=-1",
        "Plane: n={0.31623,-0.94868,0}, d=0.63246",
    };

    EXPECT_EQ(expected_node, result.node);
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    EXPECT_FALSE(result.interior) << result.interior;
    EXPECT_VEC_SOFT_EQ((Real3{-1, -1, -3}), result.exterior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{2, 1, 3}), result.exterior.upper());

    this->check_corners(result.node_id, pri, 0.1);
}

TEST_F(GenPrismTest, tetrahedron)
{
    auto pri
        = GenPrism(3, {{-1, -1}, {2, 0}, {-1, 1}}, {{0, 0}, {0, 0}, {0, 0}});

    static real_type const expected_twist_angles[] = {0, 0, 0};
    EXPECT_VEC_SOFT_EQ(expected_twist_angles, this->get_twist_angles(pri));

    auto result = this->test(pri);
    static char const expected_node[] = "all(+0, -1, -2, +3)";
    static char const* const expected_surfaces[] = {
        "Plane: z=-3",
        "Plane: n={0.31449,-0.94346,0.10483}, d=0.31449",
        "Plane: n={0.31449,0.94346,0.10483}, d=0.31449",
        "Plane: n={0.98639,0,-0.16440}, d=-0.49320",
    };

    EXPECT_EQ(expected_node, result.node);
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    EXPECT_FALSE(result.interior) << result.interior;
    EXPECT_VEC_SOFT_EQ((Real3{-1, -1, -3}), result.exterior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{2, 1, 3}), result.exterior.upper());

    this->check_corners(result.node_id, pri, 0.01);
}

TEST_F(GenPrismTest, odd_tetrahedron)
{
    auto pri
        = GenPrism(3, {{2, 0}, {2, 0}, {2, 0}}, {{-1, -1}, {2, 0}, {-1, 1}});

    static real_type const expected_twist_angles[] = {0, 0, 0};
    EXPECT_VEC_SOFT_EQ(expected_twist_angles, this->get_twist_angles(pri));

    auto result = this->test(pri);
    static char const expected_node[] = "all(-0, -1, -2, +3)";
    static char const* const expected_surfaces[] = {
        "Plane: z=3",
        "Plane: n={0.31623,-0.94868,0}, d=0.63246",
        "Plane: n={0.31623,0.94868,0}, d=0.63246",
        "Plane: n={0.89443,0,0.44721}, d=0.44721",
    };

    EXPECT_EQ(expected_node, result.node);
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    EXPECT_FALSE(result.interior) << result.interior;
    EXPECT_VEC_SOFT_EQ((Real3{-1, -1, -3}), result.exterior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{2, 1, 3}), result.exterior.upper());

    this->check_corners(result.node_id, pri, 0.01);
}

TEST_F(GenPrismTest, envelope)
{
    GenPrism pri(2,
                 {{-1, -2}, {1, -2}, {1, 2}, {-1, 2}},
                 {{-0.5, 0}, {0.5, 0}, {0.5, 0}, {-0.5, 0}});

    static real_type const expected_twist_angles[] = {0, 0, 0, 0};
    EXPECT_VEC_SOFT_EQ(expected_twist_angles, this->get_twist_angles(pri));

    auto result = this->test(pri);
    static char const expected_node[] = "all(+0, +1, -2, -3, +4)";
    static char const* const expected_surfaces[] = {
        "Plane: z=-2",
        "Plane: n={0,0.89443,-0.44721}, d=-0.89443",
        "Plane: n={0.99228,-0,0.12404}, d=0.74421",
        "Plane: n={0,0.89443,0.44721}, d=0.89443",
        "Plane: n={0.99228,0,-0.12404}, d=-0.74421",
    };

    EXPECT_EQ(expected_node, result.node);
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    EXPECT_FALSE(result.interior) << result.interior;
    EXPECT_VEC_SOFT_EQ((Real3{-1, -2, -2}), result.exterior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{1, 2, 2}), result.exterior.upper());

    this->check_corners(result.node_id, pri, 0.01);
}

TEST_F(GenPrismTest, trd)
{
    auto pri = GenPrism::from_trd(3, {1, 1}, {2, 2});

    static real_type const expected_lower[] = {1, -1, 1, 1, -1, 1, -1, -1};
    static real_type const expected_upper[] = {2, -2, 2, 2, -2, 2, -2, -2};
    EXPECT_VEC_SOFT_EQ(expected_lower, to_vec(pri.lower()));
    EXPECT_VEC_SOFT_EQ(expected_upper, to_vec(pri.upper()));

    auto result = this->test(pri);
    static char const expected_node[] = "all(+0, -1, -2, -3, +4, +5)";
    static char const* const expected_surfaces[] = {
        "Plane: z=-3",
        "Plane: z=3",
        "Plane: n={0.98639,0,-0.16440}, d=1.4796",
        "Plane: n={0,0.98639,-0.16440}, d=1.4796",
        "Plane: n={0.98639,0,0.16440}, d=-1.4796",
        "Plane: n={0,0.98639,0.16440}, d=-1.4796",
    };

    EXPECT_EQ(expected_node, result.node);
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    EXPECT_FALSE(result.interior) << result.interior;
    EXPECT_VEC_SOFT_EQ((Real3{-2, -2, -3}), result.exterior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{2, 2, 3}), result.exterior.upper());

    this->check_corners(result.node_id, pri, 0.1);
}

// Test a trapezoid used by the ATLAS LAr calorimeter geometry that has a
// zero-area polygon on the lower face
TEST_F(GenPrismTest, trd_degen)
{
    auto pri = GenPrism::from_trd(3, {0, 1}, {1, 1});
    auto result = this->test(pri);
    IntersectTestResult ref;
    ref.node = "all(-0, -1, -2, +3, +4)";
    ref.surfaces = {
        "Plane: z=3",
        "Plane: n={0.98639,0,-0.16440}, d=0.49320",
        "Plane: y=1",
        "Plane: n={0.98639,0,0.16440}, d=-0.49320",
        "Plane: y=-1",
    };
    ref.interior = {};
    ref.exterior = {{-1, -1, -3}, {1, 1, 3}};
    EXPECT_REF_EQ(ref, result);
}

TEST_F(GenPrismTest, trap_theta)
{
    auto pri = GenPrism::from_trap(
        40, Turn{0.125}, Turn{0}, {20, 10, 10, Turn{}}, {20, 10, 10, Turn{}});
    static real_type const expected_lower[]
        = {-30, -20, -30, 20, -50, 20, -50, -20};
    static real_type const expected_upper[]
        = {50, -20, 50, 20, 30, 20, 30, -20};
    EXPECT_VEC_SOFT_EQ(expected_lower, to_vec(pri.lower()));
    EXPECT_VEC_SOFT_EQ(expected_upper, to_vec(pri.upper()));

    auto result = this->test(pri);
    this->check_corners(result.node_id, pri, 1.0);
}

TEST_F(GenPrismTest, trap_thetaphi)
{
    auto pri = GenPrism::from_trap(40,
                                   Turn{0.125},
                                   Turn{0.25},
                                   {20, 10, 10, Turn{0}},
                                   {20, 10, 10, Turn{0}});
    static real_type const expected_lower[]
        = {10, -60, 10, -20, -10, -20, -10, -60};
    static real_type const expected_upper[]
        = {10, 20, 10, 60, -10, 60, -10, 20};
    EXPECT_VEC_SOFT_EQ(expected_lower, to_vec(pri.lower()));
    EXPECT_VEC_SOFT_EQ(expected_upper, to_vec(pri.upper()));

    auto result = this->test(pri);
    this->check_corners(result.node_id, pri, 1.0);
}

TEST_F(GenPrismTest, trap_g4)
{
    constexpr Turn degree{real_type{1} / 360};

    auto pri = GenPrism::from_trap(4,
                                   5 * degree,
                                   10 * degree,
                                   {2, 1, 1, 15 * degree},
                                   {3, 1.5, 1.5, 15 * degree});
    auto result = this->test(pri);
    static char const expected_node[] = "all(+0, -1, +2, -3, -4, +5)";
    static char const* const expected_surfaces[] = {
        "Plane: z=-4",
        "Plane: z=4",
        "Plane: n={-0.95664,0.25633,0.13832}, d=-1.1958",
        "Plane: n={0,0.99032,-0.13883}, d=2.4758",
        "Plane: n={-0.96575,0.25877,0.018918}, d=1.2072",
        "Plane: n={0,0.99403,0.10915}, d=-2.4851",
    };

    EXPECT_EQ(expected_node, result.node);
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    EXPECT_FALSE(result.interior) << result.interior;
    EXPECT_VEC_SOFT_EQ((Real3{-1.9592095207293, -2.9392310120488, -4}),
                       result.exterior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{2.6484856338574, 3.0607689879512, 4}),
                       result.exterior.upper());

    this->check_corners(result.node_id, pri, 0.1);
}

TEST_F(GenPrismTest, trap_full)
{
    auto pri = GenPrism::from_trap(40,
                                   Turn{0.125},
                                   Turn{0.125},
                                   {20, 10, 10, atan_to_turn(0.1)},
                                   {20, 10, 10, atan_to_turn(0.1)});

    static real_type const expected_twist_angles[] = {0, 0, 0, 0};
    EXPECT_VEC_SOFT_EQ(expected_twist_angles, this->get_twist_angles(pri));

    static real_type const expected_lower[] = {
        -20.284271247462,
        -48.284271247462,
        -16.284271247462,
        -8.2842712474619,
        -36.284271247462,
        -8.2842712474619,
        -40.284271247462,
        -48.284271247462,
    };
    static real_type const expected_upper[] = {
        36.284271247462,
        8.2842712474619,
        40.284271247462,
        48.284271247462,
        20.284271247462,
        48.284271247462,
        16.284271247462,
        8.2842712474619,
    };
    EXPECT_VEC_SOFT_EQ(expected_lower, to_vec(pri.lower()));
    EXPECT_VEC_SOFT_EQ(expected_upper, to_vec(pri.upper()));

    auto result = this->test(pri);
    this->check_corners(result.node_id, pri, 1.0);
}

TEST_F(GenPrismTest, trap_full2)
{
    auto pri = GenPrism::from_trap(40,
                                   Turn{0.125},
                                   Turn{0},
                                   {20, 10, 10, atan_to_turn(0.1)},
                                   {20, 10, 15, -atan_to_turn(0.2)});

    static real_type const expected_twist_angles[]
        = {0.027777073517552, 0, 0.065874318731703, 0};
    EXPECT_VEC_SOFT_EQ(expected_twist_angles, this->get_twist_angles(pri));

    auto result = this->test(pri);
    static char const expected_node[] = "all(+0, -1, -2, -3, +4, +5)";
    static char const* const expected_surfaces[] = {
        "Plane: z=-40",
        "Plane: z=40",
        R"(GQuadric: {0,0,0} {0,0.0015228,0} {0.69612,-0.0087015,-0.71787} -7.8313)",
        "Plane: y=20",
        "GQuadric: {0,0,0} {0,0.0038033,0} {0.71591,0.080539,-0.69354} 8.0540",
        "Plane: y=-20",
    };

    EXPECT_EQ(expected_node, result.node);
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    EXPECT_FALSE(result.interior) << result.interior;
    EXPECT_VEC_SOFT_EQ((Real3{-52, -20, -40}), result.exterior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{54, 20, 40}), result.exterior.upper());

    this->check_corners(result.node_id, pri, 1.0);
}

TEST_F(GenPrismTest, trap_quarter_twist)
{
    auto pri = GenPrism::from_trap(
        1, Turn{0}, Turn{0}, {1, 2, 2, -Turn{0.125}}, {1, 2, 2, Turn{0.125}});

    static real_type const expected_twist_angles[] = {0.25, 0, 0.25, 0};
    EXPECT_VEC_SOFT_EQ(expected_twist_angles, this->get_twist_angles(pri));

    static Real2 const expected_lower[] = {{3, -1}, {1, 1}, {-3, 1}, {-1, -1}};
    static Real2 const expected_upper[] = {{1, -1}, {3, 1}, {-1, 1}, {-3, -1}};
    EXPECT_VEC_EQ(expected_lower, pri.lower());
    EXPECT_VEC_EQ(expected_upper, pri.upper());

    auto result = this->test(pri);
    static char const expected_node[] = "all(+0, -1, +2, -3, -4, +5)";
    static char const* const expected_surfaces[] = {
        "Plane: z=-1",
        "Plane: z=1",
        "GQuadric: {0,0,0} {0,1,0} {-1,0,0} 2",
        "Plane: y=1",
        "GQuadric: {0,0,0} {0,1,0} {-1,0,0} -2",
        "Plane: y=-1",
    };

    EXPECT_EQ(expected_node, result.node);
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    EXPECT_FALSE(result.interior) << result.interior;
    EXPECT_VEC_SOFT_EQ((Real3{-3, -1, -1}), result.exterior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{3, 1, 1}), result.exterior.upper());

    this->check_corners(result.node_id, pri, 1.0);
}

TEST_F(GenPrismTest, trap_uneven_twist)
{
    auto pri = GenPrism::from_trap(
        1, Turn{0}, Turn{0}, {1, 2, 2, Turn{0}}, {0.5, 1, 1, Turn{0.125}});

    static real_type const expected_twist_angles[] = {0.125, 0, 0.125, 0};
    EXPECT_VEC_SOFT_EQ(expected_twist_angles, this->get_twist_angles(pri));

    static real_type const expected_lower[] = {2, -1, 2, 1, -2, 1, -2, -1};
    static real_type const expected_upper[]
        = {0.5, -0.5, 1.5, 0.5, -0.5, 0.5, -1.5, -0.5};
    EXPECT_VEC_SOFT_EQ(expected_lower, to_vec(pri.lower()));
    EXPECT_VEC_SOFT_EQ(expected_upper, to_vec(pri.upper()));

    auto result = this->test(pri);

    static char const expected_node[] = "all(+0, -1, +2, -3, +4, +5)";
    static char const* const expected_surfaces[] = {
        "Plane: z=-1",
        "Plane: z=1",
        R"(GQuadric: {0,0,0.11471} {0,0.22942,0.22942} {-0.68825,0.22942,-0.68825} 1.0324)",
        "Plane: n={0,0.97014,0.24254}, d=0.72761",
        R"(GQuadric: {0,0,0.11471} {0,-0.22942,-0.22942} {0.68825,-0.22942,-0.68825} 1.0324)",
        "Plane: n={0,0.97014,-0.24254}, d=-0.72761",
    };

    EXPECT_EQ(expected_node, result.node);
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    EXPECT_FALSE(result.interior) << result.interior;
    EXPECT_VEC_SOFT_EQ((Real3{-2, -1, -1}), result.exterior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{2, 1, 1}), result.exterior.upper());

    this->check_corners(result.node_id, pri, 0.1);
}

TEST_F(GenPrismTest, trap_even_twist)
{
    auto pri = GenPrism::from_trap(
        1, Turn{0}, Turn{0}, {1, 2, 2, Turn{0}}, {0.5, 1, 1, Turn{0.125}});

    static real_type const expected_twist_angles[] = {0.125, 0, 0.125, 0};
    EXPECT_VEC_SOFT_EQ(expected_twist_angles, this->get_twist_angles(pri));

    static real_type const expected_lower[] = {2, -1, 2, 1, -2, 1, -2, -1};
    static real_type const expected_upper[]
        = {0.5, -0.5, 1.5, 0.5, -0.5, 0.5, -1.5, -0.5};
    EXPECT_VEC_SOFT_EQ(expected_lower, to_vec(pri.lower()));
    EXPECT_VEC_SOFT_EQ(expected_upper, to_vec(pri.upper()));

    auto result = this->test(pri);

    static char const expected_node[] = "all(+0, -1, +2, -3, +4, +5)";
    static char const* const expected_surfaces[] = {
        "Plane: z=-1",
        "Plane: z=1",
        R"(GQuadric: {0,0,0.11471} {0,0.22942,0.22942} {-0.68825,0.22942,-0.68825} 1.0324)",
        "Plane: n={0,0.97014,0.24254}, d=0.72761",
        R"(GQuadric: {0,0,0.11471} {0,-0.22942,-0.22942} {0.68825,-0.22942,-0.68825} 1.0324)",
        "Plane: n={0,0.97014,-0.24254}, d=-0.72761",
    };
    EXPECT_EQ(expected_node, result.node);
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);

    this->check_corners(result.node_id, pri, 0.1);
}

/*!
 * Test deduplication of two opposing quadric surfaces.
 *
 * \verbatim
   Lower polygons:      Upper polygons:

   x=-1      x=1           x=-0.5
   +----+----+ y=1      +--+------+ y=1
   |    |    |          |   \     |
   |    |  R |          |    \  R |
   |  L |    |          |  L  \   |
   |    |    |          |      \  |
   +----+----+ y=-1     +-------+-+ y=-1
        x=0                     x=0.5
   \endverbatim
 */
TEST_F(GenPrismTest, adjacent_twisted)
{
    {
        // Left
        auto result
            = this->test("left",
                         GenPrism(1,
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
            = this->test("right",
                         GenPrism(1,
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
        auto result = this->test("scaled",
                                 GenPrism(1,
                                          {{0, -2}, {2, -2}, {2, 2}, {0, 2}},
                                          {{1, -2}, {2, -2}, {2, 2}, {-1, 2}}));
        static char const expected_node[] = "all(+0, -1, +3, +7, -8, -9)";

        EXPECT_EQ(expected_node, result.node);
        EXPECT_VEC_SOFT_EQ((Real3{-1, -2, -1}), result.exterior.lower());
        EXPECT_VEC_SOFT_EQ((Real3{2, 2, 1}), result.exterior.upper());
    }

    static char const* const expected_surfaces[] = {
        "Plane: z=-1",
        "Plane: z=1",
        "Plane: y=-1",
        "GQuadric: {0,0,0} {0,0.24254,0} {0.97014,0.24254,0} 0",
        "Plane: y=1",
        "Plane: x=-1",
        "Plane: x=1",
        "Plane: y=-2",
        "Plane: x=2",
        "Plane: y=2",
    };
    EXPECT_VEC_EQ(expected_surfaces, surface_strings(this->unit()));

    auto node_strings = md_strings(this->unit());
    static char const* const expected_node_strings[] = {
        "",
        "",
        "left@mz,right@mz,scaled@mz",
        "left@pz,right@pz,scaled@pz",
        "",
        "left@p0,right@p0",
        "left@t1,right@t3,scaled@t3",
        "",
        "left@p2,right@p2",
        "",
        "left@p3",
        "left",
        "right@p1",
        "",
        "right",
        "scaled@p0",
        "scaled@p1",
        "",
        "scaled@p2",
        "",
        "scaled",
    };
    EXPECT_VEC_EQ(expected_node_strings, node_strings);
}

TEST_F(GenPrismTest, emec_blade)
{
    // Reset to using "default" tolerance, 1mm length scale
    this->reset_with_tol(Tol::from_default(1.0));

    auto result = this->test(GenPrism(10.625,
                                      {{1.55857990922689, 302.468976599716},
                                       {-1.73031296208306, 302.468976599716},
                                       {-2.53451906396442, 609.918546236458},
                                       {2.18738922312177, 609.918546236458}},
                                      {{-11.9586196560814, 304.204253530802},
                                       {-15.2556006134987, 304.204253530802},
                                       {-31.2774318502685, 613.426120316623},
                                       {-26.5391748405779, 613.426120316623}}));

    if constexpr (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_FLOAT)
    {
        GTEST_SKIP()
            << "Tolerance changes with floating point type, "
               "so the GQ sign is flipped because it's ignored as zero since "
               "it's below tolerance";
    }

    static char const* const expected_surface_strings[] = {
        "Plane: z=-10.625",
        "Plane: z=10.625",
        "Plane: n={0,0.98665,-0.16286}, d=603.51",
        R"(GQuadric: {0,0,1.7449e-5} {0,-0.0023163,-0.00026977} {-0.99733,-0.027212,0.067778} -0.21576)",
        "Plane: n={0,0.99668,-0.081389}, d=302.33",
        R"(GQuadric: {0,0,1.7450e-5} {0,-0.0023153,-0.00026979} {-0.99741,-0.022566,0.068291} 1.6584)",
    };
    static char const* const expected_volume_strings[] = {
        "all(+0, -1, -2, -3, +4, +5)",
    };
    static char const* const expected_md_strings[] = {
        "",
        "",
        "cr@mz",
        "cr@pz",
        "",
        "cr@p0",
        "",
        "cr@t1",
        "",
        "cr@p2",
        "cr@t3",
        "cr",
    };

    auto& u = this->unit();
    EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
    EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
    EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
}

TEST_F(GenPrismTest, variable_twisted)
{
    using SS = SignedSense;
    char label = 'A';
    constexpr real_type x = 10;
    constexpr real_type hh = 1;
    auto build_prism = [&](real_type eps) {
        std::string const label_str(1, label++);
        SCOPED_TRACE(label_str);
        // Build and insert a node
        auto n = this->insert(
            this->build(label_str,
                        GenPrism(hh,
                                 {{x - eps, -1}, {x + eps, 1}, {0, 0}},
                                 {{x + eps, -1}, {x - eps, 1}, {0, 0}}),
                        NoTransformation{}));

        if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_FLOAT
            && label_str == "D")
        {
            // First twisted surface has small enough coefficients that the
            // corners aren't quite accurate
            return n;
        }

        // Test corners
        auto tol_eps = this->tol().rel;
        {
            SCOPED_TRACE("z = -1");
            // [lo][0]
            EXPECT_EQ(
                SS::inside,
                this->calc_sense(n, {x - eps, -1 + tol_eps, -hh + tol_eps}));
            EXPECT_EQ(SS::outside,
                      this->calc_sense(n, {x + eps, -1, -1 + tol_eps}));
            // [lo][0.5]
            EXPECT_EQ(SS::inside,
                      this->calc_sense(n, {x - tol_eps, 0, -hh + tol_eps}));
            EXPECT_EQ(SS::outside,
                      this->calc_sense(n, {x + tol_eps, 0, -hh + tol_eps}));
        }
        {
            SCOPED_TRACE("z = 0");
            // [mid][0.5]
            EXPECT_EQ(SS::inside, this->calc_sense(n, {x - tol_eps, 0, 0}));
            EXPECT_EQ(SS::outside, this->calc_sense(n, {x + tol_eps, 0, 0}));
        }
        {
            SCOPED_TRACE("z = 1");
            // [hi][1]
            EXPECT_EQ(
                SS::inside,
                this->calc_sense(n, {x - eps, 1 - tol_eps, hh - tol_eps}));
            EXPECT_EQ(
                SS::outside,
                this->calc_sense(n, {x + eps, 1 - tol_eps, hh - tol_eps}));
            // [hi][0.5]
            EXPECT_EQ(SS::inside,
                      this->calc_sense(n, {x - tol_eps, 0, hh - tol_eps}));
            EXPECT_EQ(SS::outside,
                      this->calc_sense(n, {x + tol_eps, 0, hh - tol_eps}));
        }

        return n;
    };
    for (auto logeps : range(-6, -1))
    {
        build_prism(std::pow(real_type{10}, static_cast<real_type>(logeps)));
    }
    for (auto fraceps : range(0, 5))
    {
        build_prism(0.1 + real_type{0.025} * fraceps);
    }

    auto const& u = this->unit();

    static char const* const expected_surface_strings[] = {
        "Plane: z=-1",
        "Plane: z=1",
        "Plane: x=10",
        "Plane: n={0.099504,-0.99504,0}, d=0",
        "Plane: n={0.099504,0.99504,0}, d=0",
        "GQuadric: {0,0,0} {0,1e-3,0} {1,0,0} -10",
        "GQuadric: {0,0,0} {0,0.01,0} {1,0,0} -10",
        "GQuadric: {0,0,0} {0,0.00099504,0} {0.099504,-0.99504,0} 0",
        "GQuadric: {0,0,0} {0,0.00099504,0} {0.099504,0.99504,0} 0",
        "GQuadric: {0,0,0} {0,0.1,0} {1,0,0} -10",
        "GQuadric: {0,0,0} {0,0.0099504,0} {0.099504,-0.99504,0} 0",
        "GQuadric: {0,0,0} {0,0.0099504,0} {0.099504,0.99504,0} 0",
        "GQuadric: {0,0,0} {0,0.125,0} {1,0,0} -10",
        "GQuadric: {0,0,0} {0,0.012438,0} {0.099504,-0.99504,0} 0",
        "GQuadric: {0,0,0} {0,0.012438,0} {0.099504,0.99504,0} 0",
        "GQuadric: {0,0,0} {0,0.15,0} {1,0,0} -10",
        "GQuadric: {0,0,0} {0,0.014926,0} {0.099504,-0.99504,0} 0",
        "GQuadric: {0,0,0} {0,0.014926,0} {0.099504,0.99504,0} 0",
        "GQuadric: {0,0,0} {0,0.175,0} {1,0,0} -10",
        "GQuadric: {0,0,0} {0,0.017413,0} {0.099504,-0.99504,0} 0",
        "GQuadric: {0,0,0} {0,0.017413,0} {0.099504,0.99504,0} 0",
        "GQuadric: {0,0,0} {0,0.2,0} {1,0,0} -10",
        "GQuadric: {0,0,0} {0,0.019901,0} {0.099504,-0.99504,0} 0",
        "GQuadric: {0,0,0} {0,0.019901,0} {0.099504,0.99504,0} 0",
    };
    static char const* const expected_volume_strings[] = {
        "all(+0, -1, -2, +3, +4)",
        "all(+0, -1, -2, +3, +4)",
        "all(+0, -1, -2, +3, +4)",
        "all(+0, -1, +3, +4, -5)",
        "all(+0, -1, -6, +7, +8)",
        "all(+0, -1, -9, +10, +11)",
        "all(+0, -1, -12, +13, +14)",
        "all(+0, -1, -15, +16, +17)",
        "all(+0, -1, -18, +19, +20)",
        "all(+0, -1, -21, +22, +23)",
    };
    static char const* const expected_md_strings[] = {
        "",
        "",
        "A@mz,B@mz,C@mz,D@mz,E@mz,F@mz,G@mz,H@mz,I@mz,J@mz",
        "A@pz,B@pz,C@pz,D@pz,E@pz,F@pz,G@pz,H@pz,I@pz,J@pz",
        "",
        "A@p0,B@p0,C@p0",
        "",
        "A@p1,B@p1,C@p1,D@p1",
        "A@p2,B@p2,C@p2,D@p2",
        "A,B,C",
        "D@t0",
        "",
        "D",
        "E@t0",
        "",
        "E@t1",
        "E@t2",
        "E",
        "F@t0",
        "",
        "F@t1",
        "F@t2",
        "F",
        "G@t0",
        "",
        "G@t1",
        "G@t2",
        "G",
        "H@t0",
        "",
        "H@t1",
        "H@t2",
        "H",
        "I@t0",
        "",
        "I@t1",
        "I@t2",
        "I",
        "J@t0",
        "",
        "J@t1",
        "J@t2",
        "J",
    };

    if constexpr (CELERITAS_REAL_TYPE != CELERITAS_REAL_TYPE_FLOAT)
    {
        // Slight changes in gquadric construction
        EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
    }
    EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
    EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
}

//---------------------------------------------------------------------------//
// HYPERBOLOID
//---------------------------------------------------------------------------//
using HyperboloidTest = IntersectRegionTest;

TEST_F(HyperboloidTest, errors)
{
    // Negative middle radius
    EXPECT_THROW(Hyperboloid(-1, 2, 1), RuntimeError);
    // Zero middle radius
    EXPECT_THROW(Hyperboloid(0, 2, 1), RuntimeError);
    // Negative top radius
    EXPECT_THROW(Hyperboloid(1, -2, 1), RuntimeError);
    // Top radius not greater than middle radius (equal)
    EXPECT_THROW(Hyperboloid(2, 2, 1), RuntimeError);
    // Top radius less than middle radius
    EXPECT_THROW(Hyperboloid(2, 1, 1), RuntimeError);
    // Negative half-height
    EXPECT_THROW(Hyperboloid(1, 2, -1), RuntimeError);
    // Zero half-height
    EXPECT_THROW(Hyperboloid(1, 2, 0), RuntimeError);
}

TEST_F(HyperboloidTest, standard)
{
    auto result = this->test(Hyperboloid(1, 2, 3));

    IntersectTestResult ref;
    ref.node = "all(+0, -1, -2)";
    ref.surfaces
        = {"Plane: z=-3", "Plane: z=3", "SQuadric: {1,1,-0.33333} {0,0,0} -1"};
    ref.interior = {{-0.70710678118655, -0.70710678118655, -3},
                    {0.70710678118655, 0.70710678118655, 3}};
    ref.exterior = {{-2, -2, -3}, {2, 2, 3}};
    EXPECT_REF_EQ(ref, result);
}

//---------------------------------------------------------------------------//
// INFPLANE
//---------------------------------------------------------------------------//
using InfPlaneTest = IntersectRegionTest;

TEST_F(InfPlaneTest, basic)
{
    using Plane = InfPlane;

    auto inf = std::numeric_limits<real_type>::infinity();
    {
        auto result = this->test(Plane(Sense::inside, Axis::x, -1.5));
        IntersectTestResult ref;
        ref.node = "-0";
        ref.surfaces = {"Plane: x=-1.5"};
        ref.interior = {{-inf, -inf, -inf}, {-1.5, inf, inf}};
        ref.exterior = {{-inf, -inf, -inf}, {-1.5, inf, inf}};
        EXPECT_REF_EQ(ref, result);
    }
    {
        auto result = this->test(Plane(Sense::outside, Axis::z, 2));

        IntersectTestResult ref;
        ref.node = "+1";
        ref.surfaces = {"Plane: x=-1.5", "Plane: z=2"};
        ref.interior = {{-inf, -inf, 2}, {inf, inf, inf}};
        ref.exterior = {{-inf, -inf, 2}, {inf, inf, inf}};
        EXPECT_REF_EQ(ref, result);
    }
}

//---------------------------------------------------------------------------//
// INFAZIWEDGE
//---------------------------------------------------------------------------//
using InfAziWedgeTest = IntersectRegionTest;

TEST_F(InfAziWedgeTest, errors)
{
    EXPECT_THROW(InfAziWedge(Turn{0}, Turn{0.51}), RuntimeError);
    EXPECT_THROW(InfAziWedge(Turn{0}, Turn{0}), RuntimeError);
    EXPECT_THROW(InfAziWedge(Turn{0}, Turn{-0.5}), RuntimeError);
    EXPECT_THROW(InfAziWedge(Turn{-0.1}, Turn{-0.5}), RuntimeError);
    EXPECT_THROW(InfAziWedge(Turn{1.1}, Turn{-0.5}), RuntimeError);
}

TEST_F(InfAziWedgeTest, quarter_turn)
{
    auto inf = std::numeric_limits<real_type>::infinity();
    {
        SCOPED_TRACE("first quadrant");
        auto result = this->test(InfAziWedge(Turn{0}, Turn{0.25}));
        static char const expected_node[] = "all(+0, +1)";
        static char const* const expected_surfaces[]
            = {"Plane: x=0", "Plane: y=0"};

        EXPECT_EQ(expected_node, result.node);
        EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
        EXPECT_VEC_SOFT_EQ((Real3{0, 0, -inf}), result.interior.lower());
        EXPECT_VEC_SOFT_EQ((Real3{inf, inf, inf}), result.interior.upper());
        EXPECT_VEC_SOFT_EQ((Real3{0, 0, -inf}), result.exterior.lower());
        EXPECT_VEC_SOFT_EQ((Real3{inf, inf, inf}), result.exterior.upper());
    }
    {
        SCOPED_TRACE("second quadrant");
        auto result = this->test(InfAziWedge(Turn{.25}, Turn{0.5}));
        EXPECT_EQ("all(+1, -0)", result.node);
    }
    {
        SCOPED_TRACE("fourth quadrant");
        InfAziWedge wedge(Turn{0.75}, Turn{1.0});
        EXPECT_SOFT_EQ(0.75, wedge.start().value());
        auto result = this->test(wedge);
        EXPECT_EQ("all(+0, -1)", result.node);
    }
    {
        SCOPED_TRACE("north quadrant");
        auto result = this->test(InfAziWedge(Turn{0.125}, Turn{0.375}));
        EXPECT_EQ("all(+2, -3)", result.node);
    }
    {
        SCOPED_TRACE("east quadrant");
        auto result = this->test(InfAziWedge(Turn{0.875}, Turn{1.125}));
        EXPECT_EQ("all(+2, +3)", result.node);
        static char const expected_node[] = "all(+2, +3)";
        EXPECT_EQ(expected_node, result.node);
        EXPECT_FALSE(result.interior) << result.interior;
        EXPECT_EQ(BBox::from_infinite(), result.exterior);
    }
    {
        SCOPED_TRACE("west quadrant");
        auto result = this->test(InfAziWedge(Turn{0.375}, Turn{0.625}));
        static char const expected_node[] = "all(-3, -2)";
        static char const* const expected_surfaces[] = {
            "Plane: x=0",
            "Plane: y=0",
            "Plane: n={0.70711,0.70711,0}, d=0",
            "Plane: n={0.70711,-0.70711,0}, d=0",
        };

        EXPECT_EQ(expected_node, result.node);
        EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    }
}

TEST_F(InfAziWedgeTest, half_turn)
{
    auto inf = std::numeric_limits<real_type>::infinity();
    {
        SCOPED_TRACE("north half");
        auto result = this->test(InfAziWedge(Turn{0}, Turn{0.5}));
        EXPECT_EQ("+0", result.node);
        EXPECT_VEC_SOFT_EQ((Real3{-inf, 0, -inf}), result.interior.lower());
        EXPECT_VEC_SOFT_EQ((Real3{inf, inf, inf}), result.interior.upper());
        EXPECT_VEC_SOFT_EQ((Real3{-inf, 0, -inf}), result.exterior.lower());
        EXPECT_VEC_SOFT_EQ((Real3{inf, inf, inf}), result.exterior.upper());
    }
    {
        SCOPED_TRACE("south half");
        auto result = this->test(InfAziWedge(Turn{0.5}, Turn{1.0}));
        EXPECT_EQ("-0", result.node);
    }
    {
        SCOPED_TRACE("northeast half");
        auto result = this->test(InfAziWedge(Turn{0.125}, Turn{0.625}));
        static char const expected_node[] = "-1";
        static char const* const expected_surfaces[]
            = {"Plane: y=0", "Plane: n={0.70711,-0.70711,0}, d=0"};

        EXPECT_EQ(expected_node, result.node);
        EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    }
}
//---------------------------------------------------------------------------//
// INFPOLARWEDGE
//---------------------------------------------------------------------------//
using InfPolarWedgeTest = IntersectRegionTest;

TEST_F(InfPolarWedgeTest, errors)
{
    EXPECT_THROW(InfPolarWedge(Turn{-0.2}, Turn{-0.001}), RuntimeError);
    EXPECT_THROW(InfPolarWedge(Turn{-0.1}, Turn{0.1}), RuntimeError);
    EXPECT_THROW(InfPolarWedge(Turn{0}, Turn{-0.1}), RuntimeError);
    EXPECT_THROW(InfPolarWedge(Turn{0}, Turn{0.26}), RuntimeError);
    EXPECT_THROW(InfPolarWedge(Turn{0.1}, Turn{0.1}), RuntimeError);
    EXPECT_THROW(InfPolarWedge(Turn{0.24}, Turn{0.26}), RuntimeError);
    EXPECT_THROW(InfPolarWedge(Turn{0.26}, Turn{0.52}), RuntimeError);
}

TEST_F(InfPolarWedgeTest, quarter_turn)
{
    auto inf = std::numeric_limits<real_type>::infinity();
    {
        SCOPED_TRACE("top half");
        auto result = this->test(InfPolarWedge(Turn{0}, Turn{0.25}));
        IntersectTestResult ref;
        ref.node = "+0";
        ref.surfaces = {"Plane: z=0"};
        ref.interior = {{-inf, -inf, 0}, {inf, inf, inf}};
        ref.exterior = {{-inf, -inf, 0}, {inf, inf, inf}};
        EXPECT_REF_EQ(ref, result);
    }
    {
        SCOPED_TRACE("bottom half");
        auto result = this->test(InfPolarWedge(Turn{0.25}, Turn{0.5}));
        IntersectTestResult ref;
        ref.node = "-0";
        ref.surfaces = {"Plane: z=0"};
        ref.interior = {{-inf, -inf, -inf}, {inf, inf, 0}};
        ref.exterior = {{-inf, -inf, -inf}, {inf, inf, 0}};
        EXPECT_REF_EQ(ref, result);
    }
}

TEST_F(InfPolarWedgeTest, eighth_turn)
{
    auto inf = std::numeric_limits<real_type>::infinity();
    {
        SCOPED_TRACE("north pole");
        auto result = this->test(InfPolarWedge(Turn{0}, Turn{0.125}));
        IntersectTestResult ref;
        ref.node = "all(+0, -1)";
        ref.surfaces = {"Plane: z=0", "Cone z: t=1 at {0,0,0}"};
        ref.interior = {};
        ref.exterior = {{-inf, -inf, 0}, {inf, inf, inf}};
        EXPECT_REF_EQ(ref, result);
    }
    {
        SCOPED_TRACE("north tropic");
        auto result = this->test(InfPolarWedge(Turn{0.125}, Turn{0.25}));
        IntersectTestResult ref;
        ref.node = "all(+0, +1)";
        ref.surfaces = {"Plane: z=0", "Cone z: t=1 at {0,0,0}"};
        ref.interior = {};
        ref.exterior = {{-inf, -inf, 0}, {inf, inf, inf}};
        EXPECT_REF_EQ(ref, result);
    }
    {
        SCOPED_TRACE("south tropic");
        auto result = this->test(InfPolarWedge(Turn{0.25}, Turn{0.375}));
        IntersectTestResult ref;
        ref.node = "all(+1, -0)";
        ref.surfaces = {"Plane: z=0", "Cone z: t=1 at {0,0,0}"};
        ref.interior = {};
        ref.exterior = {{-inf, -inf, -inf}, {inf, inf, 0}};
        EXPECT_REF_EQ(ref, result);
    }
    {
        SCOPED_TRACE("south pole");
        auto result = this->test(InfPolarWedge(Turn{0.375}, Turn{0.5}));
        IntersectTestResult ref;
        ref.node = "all(-1, -0)";
        ref.surfaces = {"Plane: z=0", "Cone z: t=1 at {0,0,0}"};
        ref.interior = {};
        ref.exterior = {{-inf, -inf, -inf}, {inf, inf, 0}};
        EXPECT_REF_EQ(ref, result);
    }
}

TEST_F(InfPolarWedgeTest, sliver)
{
    auto inf = std::numeric_limits<real_type>::infinity();
    {
        SCOPED_TRACE("north");
        auto result = this->test(InfPolarWedge(Turn{0.0625}, Turn{0.125}));
        IntersectTestResult ref;
        ref.node = "all(+0, +1, -2)";
        ref.surfaces = {
            "Plane: z=0",
            "Cone z: t=0.41421 at {0,0,0}",
            "Cone z: t=1 at {0,0,0}",
        };
        ref.interior = {};
        ref.exterior = {{-inf, -inf, 0}, {inf, inf, inf}};
        EXPECT_REF_EQ(ref, result);
    }
    {
        SCOPED_TRACE("south");
        auto result = this->test(InfPolarWedge(Turn{0.375}, Turn{0.4375}));
        IntersectTestResult ref;
        ref.node = "all(+1, -2, -0)";
        ref.surfaces = {
            "Plane: z=0",
            "Cone z: t=0.41421 at {0,0,0}",
            "Cone z: t=1 at {0,0,0}",
        };
        ref.interior = {};
        ref.exterior = {{-inf, -inf, -inf}, {inf, inf, 0}};
        EXPECT_REF_EQ(ref, result);
    }
}

//---------------------------------------------------------------------------//
// INVOLUTE
//---------------------------------------------------------------------------//
class InvoluteTest : public IntersectRegionTest
{
  public:
    inline static constexpr auto ccw = Chirality::left;
    inline static constexpr auto cw = Chirality::right;
};

TEST_F(InvoluteTest, single)
{
    {
        // involute
        auto result = this->test(
            "invo",
            Involute({1.0, 2.0, 4.0}, {0, 0.15667 * constants::pi}, cw, 1.0));

        static char const expected_node[] = "all(+0, -1, +2, -3, +4, -5)";

        EXPECT_EQ(expected_node, result.node);
        EXPECT_VEC_SOFT_EQ((Real3{-4, -4, -1}), result.exterior.lower());
        EXPECT_VEC_SOFT_EQ((Real3{4, 4, 1}), result.exterior.upper());
    }

    static char const* const expected_surfaces[] = {
        "Plane: z=-1",
        "Plane: z=1",
        "Cyl z: r=2",
        "Cyl z: r=4",
        "Involute cw: r=1, a=0, t={1.7321,4.3652} at x=0, y=0",
        "Involute cw: r=1, a=0.49219, t={1.7321,4.3652} at x=0, y=0",
    };
    EXPECT_VEC_EQ(expected_surfaces, surface_strings(this->unit()));

    auto node_strings = md_strings(this->unit());
    static char const* const expected_node_strings[] = {
        "",
        "",
        "invo@mz",
        "invo@pz",
        "",
        "invo@cz",
        "invo@cz",
        "",
        "invo@invl",
        "invo@invr",
        "",
        "invo",
    };
    EXPECT_VEC_EQ(expected_node_strings, node_strings);
}

// Counterclockwise adjacent involutes
TEST_F(InvoluteTest, two_ccw)
{
    {
        // involute
        auto result = this->test(
            "top",
            Involute({1.0, 2.0, 4.0}, {0, 0.15667 * constants::pi}, ccw, 1.0));

        static char const expected_node[] = "all(+0, -1, +2, -3, -4, +5)";

        EXPECT_EQ(expected_node, result.node);
        EXPECT_VEC_SOFT_EQ((Real3{-4, -4, -1}), result.exterior.lower());
        EXPECT_VEC_SOFT_EQ((Real3{4, 4, 1}), result.exterior.upper());
    }
    {
        // bottom
        auto result = this->test(
            "bottom",
            Involute({1.0, 2.0, 4.0},
                     {0.15667 * constants::pi, 0.31334 * constants::pi},
                     ccw,
                     1.0));

        static char const expected_node[] = "all(+0, -1, +2, -3, -5, +6)";
        EXPECT_EQ(expected_node, result.node);

        EXPECT_VEC_SOFT_EQ((Real3{-4, -4, -1}), result.exterior.lower());
        EXPECT_VEC_SOFT_EQ((Real3{4, 4, 1}), result.exterior.upper());
    }

    static char const* const expected_surfaces[] = {
        "Plane: z=-1",
        "Plane: z=1",
        "Cyl z: r=2",
        "Cyl z: r=4",
        "Involute ccw: r=1, a=0, t={1.7321,4.3652} at x=0, y=0",
        "Involute ccw: r=1, a=0.49219, t={1.7321,4.3652} at x=0, y=0",
        "Involute ccw: r=1, a=0.98439, t={1.7321,4.3652} at x=0, y=0",
    };
    EXPECT_VEC_EQ(expected_surfaces, surface_strings(this->unit()));

    auto node_strings = md_strings(this->unit());
    static char const* const expected_node_strings[] = {
        "",
        "",
        "bottom@mz,top@mz",
        "bottom@pz,top@pz",
        "",
        "bottom@cz,top@cz",
        "bottom@cz,top@cz",
        "",
        "top@invl",
        "",
        "bottom@invl,top@invr",
        "top",
        "",
        "bottom@invr",
        "bottom",
    };
    EXPECT_VEC_EQ(expected_node_strings, node_strings);
}

// Clockwise variant of previous
TEST_F(InvoluteTest, two_cw)
{
    {
        // involute
        auto result = this->test(
            "top",
            Involute({1.0, 2.0, 4.0}, {0, 0.15667 * constants::pi}, cw, 1.0));

        static char const expected_node[] = "all(+0, -1, +2, -3, +4, -5)";

        EXPECT_EQ(expected_node, result.node);
        EXPECT_VEC_SOFT_EQ((Real3{-4, -4, -1}), result.exterior.lower());
        EXPECT_VEC_SOFT_EQ((Real3{4, 4, 1}), result.exterior.upper());
    }
    {
        // bottom
        auto result = this->test(
            "bottom",
            Involute({1.0, 2.0, 4.0},
                     {0.15667 * constants::pi, 0.31334 * constants::pi},
                     cw,
                     1.0));

        static char const expected_node[] = "all(+0, -1, +2, -3, +5, -6)";
        EXPECT_EQ(expected_node, result.node);

        EXPECT_VEC_SOFT_EQ((Real3{-4, -4, -1}), result.exterior.lower());
        EXPECT_VEC_SOFT_EQ((Real3{4, 4, 1}), result.exterior.upper());
    }

    static char const* const expected_surfaces[] = {
        "Plane: z=-1",
        "Plane: z=1",
        "Cyl z: r=2",
        "Cyl z: r=4",
        "Involute cw: r=1, a=0, t={1.7321,4.3652} at x=0, y=0",
        "Involute cw: r=1, a=0.49219, t={1.7321,4.3652} at x=0, y=0",
        "Involute cw: r=1, a=0.98439, t={1.7321,4.3652} at x=0, y=0",
    };
    EXPECT_VEC_EQ(expected_surfaces, surface_strings(this->unit()));

    auto node_strings = md_strings(this->unit());
    static char const* const expected_node_strings[] = {
        "",
        "",
        "bottom@mz,top@mz",
        "bottom@pz,top@pz",
        "",
        "bottom@cz,top@cz",
        "bottom@cz,top@cz",
        "",
        "top@invl",
        "bottom@invl,top@invr",
        "",
        "top",
        "bottom@invr",
        "",
        "bottom",
    };
    EXPECT_VEC_EQ(expected_node_strings, node_strings);
}
//---------------------------------------------------------------------------//
// PARABOLOID
//---------------------------------------------------------------------------//
using ParaboloidTest = IntersectRegionTest;

TEST_F(ParaboloidTest, errors)
{
    // Negatives
    EXPECT_THROW(Paraboloid(-1, 3, 2), RuntimeError);
    EXPECT_THROW(Paraboloid(1, -3, 2), RuntimeError);
    EXPECT_THROW(Paraboloid(-1, -3, 2), RuntimeError);
    EXPECT_THROW(Paraboloid(1, 3, -2), RuntimeError);

    // Both zeros
    EXPECT_THROW(Paraboloid(0, 0, 2), RuntimeError);

    // Cylinder
    EXPECT_THROW(Paraboloid(5, 5, 2), RuntimeError);
}

TEST_F(ParaboloidTest, encloses)
{
    Paraboloid ec(2, 3, 5);

    EXPECT_TRUE(ec.encloses(Paraboloid(1, 1.5, 4.9)));
    EXPECT_FALSE(ec.encloses(Paraboloid(1, 1.5, 5.9)));
    EXPECT_FALSE(ec.encloses(Paraboloid(2, 3, 4.9)));
    EXPECT_TRUE(ec.encloses(Paraboloid(1.5, 2.5, 4.9)));
}

TEST_F(ParaboloidTest, standard)
{
    auto result = this->test(Paraboloid(1, 2, 3));

    static char const expected_node[] = "all(+0, -1, -2)";
    static char const* const expected_surfaces[]
        = {"Plane: z=-3", "Plane: z=3", "SQuadric: {1,1,0} {0,0,-0.5} -2.5"};

    EXPECT_EQ(expected_node, result.node);
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);

    EXPECT_VEC_SOFT_EQ((Real3{-2, -2, -3}), result.exterior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{2, 2, 3}), result.exterior.upper());
}

TEST_F(ParaboloidTest, vertex)
{
    // Vertex on upper boundary
    auto result = this->test(Paraboloid(5, 0, 5));

    static char const expected_node[] = "all(+0, -1, -2)";
    static char const* const expected_surfaces[]
        = {"Plane: z=-5", "Plane: z=5", "SQuadric: {1,1,0} {0,0,2.5} -12.5"};

    EXPECT_EQ(expected_node, result.node);
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);

    EXPECT_VEC_SOFT_EQ((Real3{-5, -5, -5}), result.exterior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{5, 5, 5}), result.exterior.upper());
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
    static char const* const expected_surfaces[] = {
        "Plane: z=-3",
        "Plane: z=3",
        "Plane: y=-2",
        "Plane: y=2",
        "Plane: x=-1",
        "Plane: x=1",
    };

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
    static char const* const expected_surfaces[] = {
        "Plane: z=-3",
        "Plane: z=3",
        "Plane: y=-1.6180",
        "Plane: y=1.6180",
        "Plane: n={0.80902,-0.58779,0}, d=-0.80902",
        "Plane: n={0.80902,-0.58779,0}, d=0.80902",
    };

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
    static char const* const expected_surfaces[] = {
        "Plane: z=-3",
        "Plane: z=3",
        "Plane: y=-2",
        "Plane: y=2",
        "Plane: n={0.80902,0,-0.58779}, d=-0.80902",
        "Plane: n={0.80902,0,-0.58779}, d=0.80902",
    };

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
    static char const* const expected_surfaces[] = {
        "Plane: z=-3",
        "Plane: z=3",
        "Plane: n={0,0.96714,-0.25423}, d=-1.5649",
        "Plane: n={0,0.96714,-0.25423}, d=1.5649",
        "Plane: n={0.80902,-0.58779,0}, d=-0.80902",
        "Plane: n={0.80902,-0.58779,0}, d=0.80902",
    };

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
    static char const expected_node[] = "all(+0, -1, -2, +3, -4)";
    static char const* const expected_surfaces[] = {
        "Plane: z=-1.2",
        "Plane: z=1.2",
        "Plane: n={0.5,0.86603,0}, d=1",
        "Plane: x=-1",
        "Plane: n={0.5,-0.86603,0}, d=1",
    };

    EXPECT_EQ(expected_node, result.node);
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    EXPECT_VEC_SOFT_EQ((Real3{-1, -1, -1.2}), result.interior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{1, 1, 1.2}), result.interior.upper());
    EXPECT_VEC_SOFT_EQ((Real3{-1, -2, -1.2}), result.exterior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{2, 2, 1.2}), result.exterior.upper());
}

TEST_F(PrismTest, rtriangle)
{
    auto result = this->test(Prism(3, 1.0, 1.2, 0.5));
    static char const expected_node[] = "all(+0, -1, -2, +3, +4)";
    static char const* const expected_surfaces[] = {
        "Plane: z=-1.2",
        "Plane: z=1.2",
        "Plane: x=1",
        "Plane: n={0.5,-0.86603,0}, d=-1",
        "Plane: n={0.5,0.86603,0}, d=-1",
    };

    EXPECT_EQ(expected_node, result.node);
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    EXPECT_VEC_SOFT_EQ((Real3{-1, -1, -1.2}), result.interior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{1, 1, 1.2}), result.interior.upper());
    EXPECT_VEC_SOFT_EQ((Real3{-2, -2, -1.2}), result.exterior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{1, 2, 1.2}), result.exterior.upper());
}

TEST_F(PrismTest, square)
{
    auto result = this->test(Prism(4, 1.0, 2.0, 0.0));
    static char const expected_node[] = "all(+0, -1, -2, +3, +4, -5)";
    static char const* const expected_surfaces[] = {
        "Plane: z=-2",
        "Plane: z=2",
        "Plane: n={0.70711,0.70711,0}, d=1",
        "Plane: n={0.70711,-0.70711,0}, d=-1",
        "Plane: n={0.70711,0.70711,0}, d=-1",
        "Plane: n={0.70711,-0.70711,0}, d=1",
    };

    EXPECT_EQ(expected_node, result.node);
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    EXPECT_VEC_SOFT_EQ((Real3{-1, -1, -2}), result.interior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{1, 1, 2}), result.interior.upper());
    EXPECT_VEC_SOFT_EQ((Real3{-1.4142135623731, -1.4142135623731, -2}),
                       result.exterior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{1.4142135623731, 1.4142135623731, 2}),
                       result.exterior.upper());
}

TEST_F(PrismTest, hex)
{
    auto result = this->test(Prism(6, 1.0, 2.0, 0.0));
    static char const expected_node[] = "all(+0, -1, -2, -3, +4, +5, +6, -7)";
    static char const* const expected_surfaces[] = {
        "Plane: z=-2",
        "Plane: z=2",
        "Plane: n={0.86603,0.5,0}, d=1",
        "Plane: y=1",
        "Plane: n={0.86603,-0.5,0}, d=-1",
        "Plane: n={0.86603,0.5,0}, d=-1",
        "Plane: y=-1",
        "Plane: n={0.86603,-0.5,0}, d=1",
    };

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
    static char const* const expected_surfaces[] = {
        "Plane: z=-2",
        "Plane: z=2",
        "Plane: x=1",
        "Plane: n={0.5,0.86603,0}, d=1",
        "Plane: n={0.5,-0.86603,0}, d=-1",
        "Plane: x=-1",
        "Plane: n={0.5,0.86603,0}, d=-1",
        "Plane: n={0.5,-0.86603,0}, d=1",
    };

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
// TET
//---------------------------------------------------------------------------//
using TetTest = IntersectRegionTest;

TEST_F(TetTest, errors)
{
    // Coplanar vertices (all in xy plane)
    EXPECT_THROW(Tet({0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}), RuntimeError);
    // Degenerate: duplicate vertices
    EXPECT_THROW(Tet({0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 0}), RuntimeError);
    // Three collinear points
    EXPECT_THROW(Tet({0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {0, 1, 1}), RuntimeError);
}

TEST_F(TetTest, standard)
{
    // Regular tetrahedron vertices
    Tet tet({1, 1, 1}, {1, -1, -1}, {-1, 1, -1}, {-1, -1, 1});

    auto result = this->test(tet);

    IntersectTestResult ref;
    ref.node = "all(-0, -1, -2, +3)";
    ref.surfaces = {
        "Plane: n={0.57735,0.57735,-0.57735}, d=0.57735",
        "Plane: n={0.57735,-0.57735,0.57735}, d=0.57735",
        "Plane: n={-0.57735,0.57735,0.57735}, d=0.57735",
        "Plane: n={0.57735,0.57735,0.57735}, d=-0.57735",
    };
    ref.interior = {};
    ref.exterior = {{-1, -1, -1}, {1, 1, 1}};
    EXPECT_REF_EQ(ref, result);

    // Test senses
    EXPECT_EQ(SignedSense::inside,
              this->calc_sense(result.node_id, Real3{0, 0, 0}));
    for (auto i : range(4))
    {
        EXPECT_EQ(SignedSense::on,
                  this->calc_sense(result.node_id, tet.vertex(i)));
    }
    EXPECT_EQ(SignedSense::outside,
              this->calc_sense(result.node_id, Real3{2, 2, 2}));
}

TEST_F(TetTest, reordered)
{
    // Right-angled tetrahedron at origin, with first two points switched
    Tet tet({1, 0, 0}, {0, 0, 0}, {0, 1, 0}, {0, 0, 1});

    auto result = this->test(tet);
    IntersectTestResult ref;
    ref.node = "all(+0, +1, -2, +3)";
    ref.surfaces = {
        "Plane: z=0",
        "Plane: y=0",
        "Plane: n={0.57735,0.57735,0.57735}, d=0.57735",
        "Plane: x=0",
    };
    ref.interior = {};
    ref.exterior = {{0, 0, 0}, {1, 1, 1}};
    EXPECT_REF_EQ(ref, result);

    EXPECT_EQ(SignedSense::inside,
              this->calc_sense(result.node_id, Real3{0.3, 0.3, 0.3}));
}

TEST_F(TetTest, soft_degenerate)
{
    Tet tet({1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, 0, 1e-6});

    auto result = this->test(tet);
    IntersectTestResult ref;
    ref.node = "F";
    ref.surfaces = {"Plane: z=0", "Plane: y=0"};
    ref.interior = {{-1, 0, 0}, {1, 1, 0}};
    ref.exterior = {{-1, 0, 0}, {1, 1, 0}};
    EXPECT_REF_EQ(ref, result);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace orangeinp
}  // namespace celeritas
