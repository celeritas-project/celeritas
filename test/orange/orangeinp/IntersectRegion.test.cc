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
auto IntersectRegionTest::test(std::string&& name,
                               IntersectRegionInterface const& r,
                               VariantTransform const& trans) -> TestResult
{
    detail::IntersectSurfaceState css;
    css.transform = &trans;
    css.make_face_name = {};
    css.object_name = std::move(name);

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

        real_type const z[] = {-pri.halfheight(), pri.halfheight()};

        for (auto i : range(2))
        {
            auto const& points = (i == 0 ? pri.lower() : pri.upper());
            for (Real2 const& p : points)
            {
                Real3 const corner{p[0], p[1], z[i]};
                auto outward = make_unit_vector(corner);

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
    // Validate contruction parameters
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
    EXPECT_THROW(GenPrism::from_trd(-3, {1, 1}, {2, 2}), RuntimeError);  // dZ<0
    EXPECT_THROW(GenPrism::from_trd(3, {-1, 1}, {2, 2}), RuntimeError);  // hx1<0
    EXPECT_THROW(GenPrism::from_trd(3, {1, -1}, {2, 2}), RuntimeError);  // hy1<0
    EXPECT_THROW(GenPrism::from_trd(3, {1, 1}, {-2, 2}), RuntimeError);  // hx2<0
    EXPECT_THROW(GenPrism::from_trd(3, {1, 1}, {2, -2}), RuntimeError);  // hy2<0

    // Trap angles are invalid (note that we do *not* have the restriction of
    // Geant4 that the turns be the same: this just ends up creating a GenTrap
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

TEST_F(GenPrismTest, trapezoid_ccw)
{
    auto pri = GenPrism(40,
                        {{-19, -30}, {21, -30}, {21, 30}, {-19, 30}},
                        {{-21, -30}, {19, -30}, {19, 30}, {-21, 30}});
    auto result = this->test(pri);

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
    static char const expected_node[] = "all(+0, -1, +2, -3, -4, +5)";
    static char const* const expected_surfaces[] = {
        "Plane: z=-4",
        "Plane: z=4",
        "GQuadric: {0,0,0} {1,-1,0} {-28,-4,-4} 48",
        "Plane: n={0,0.99228,0.12403}, d=1.4884",
        "GQuadric: {0,0,0} {1,1,0} {-28,4,4} -48",
        "Plane: y=-2",
    };

    EXPECT_EQ(expected_node, result.node);
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    EXPECT_FALSE(result.interior) << result.interior;
    EXPECT_VEC_SOFT_EQ((Real3{-2, -2, -4}), result.exterior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{2, 2, 4}), result.exterior.upper());
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
        "Plane: n={0.31623,0.94868,-0}, d=0.63246",
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
    static char const* const expected_surfaces[]
        = {"Plane: z=-3",
           "Plane: n={0.31449,-0.94346,0.10483}, d=0.31449",
           "Plane: n={0.31449,0.94346,0.10483}, d=0.31449",
           "Plane: n={0.98639,0,-0.1644}, d=-0.4932"};

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
    static char const* const expected_surfaces[]
        = {"Plane: z=3",
           "Plane: n={0.31623,-0.94868,0}, d=0.63246",
           "Plane: n={0.31623,0.94868,0}, d=0.63246",
           "Plane: n={0.89443,0,0.44721}, d=0.44721"};

    EXPECT_EQ(expected_node, result.node);
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    EXPECT_FALSE(result.interior) << result.interior;
    EXPECT_VEC_SOFT_EQ((Real3{-1, -1, -3}), result.exterior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{2, 1, 3}), result.exterior.upper());
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
    static char const* const expected_surfaces[]
        = {"Plane: z=-2",
           "Plane: n={0,0.89443,-0.44721}, d=-0.89443",
           "Plane: n={0.99228,-0,0.12403}, d=0.74421",
           "Plane: n={0,0.89443,0.44721}, d=0.89443",
           "Plane: n={0.99228,0,-0.12403}, d=-0.74421"};

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
    static char const* const expected_surfaces[]
        = {"Plane: z=-3",
           "Plane: z=3",
           "Plane: n={0.98639,0,-0.1644}, d=1.4796",
           "Plane: n={0,0.98639,-0.1644}, d=1.4796",
           "Plane: n={0.98639,0,0.1644}, d=-1.4796",
           "Plane: n={0,0.98639,0.1644}, d=-1.4796"};

    EXPECT_EQ(expected_node, result.node);
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    EXPECT_FALSE(result.interior) << result.interior;
    EXPECT_VEC_SOFT_EQ((Real3{-2, -2, -3}), result.exterior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{2, 2, 3}), result.exterior.upper());

    this->check_corners(result.node_id, pri, 0.1);
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
    static char const* const expected_surfaces[]
        = {"Plane: z=-4",
           "Plane: z=4",
           "Plane: n={-0.95664,0.25633,0.13832}, d=-1.1958",
           "Plane: n={0,0.99032,-0.13883}, d=2.4758",
           "Plane: n={-0.96575,0.25877,0.018918}, d=1.2072",
           "Plane: n={0,0.99403,0.10915}, d=-2.4851"};

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
        "GQuadric: {0,0,0} {0,7,0} {3200,-40,-3300} -36000",
        "Plane: y=20",
        "GQuadric: {0,0,0} {0,17,0} {3200,360,-3100} 36000",
        "Plane: y=-20",
    };

    EXPECT_EQ(expected_node, result.node);
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    EXPECT_FALSE(result.interior) << result.interior;
    EXPECT_VEC_SOFT_EQ((Real3{-52, -20, -40}), result.exterior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{54, 20, 40}), result.exterior.upper());
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
        "GQuadric: {0,0,0} {0,4,0} {-4,0,0} 8",
        "Plane: y=1",
        "GQuadric: {0,0,0} {0,4,0} {-4,0,0} -8",
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

    static char const expected_node[] = "all(+0, -1, +2, -3, -4, +5)";
    static char const* const expected_surfaces[] = {
        "Plane: z=-1",
        "Plane: z=1",
        "GQuadric: {0,0,0} {1,1,0} {-3,1,-3} 5",
        "Plane: n={0,0.97014,0.24254}, d=0.72761",
        "GQuadric: {0,0,0} {1,1,0} {-3,1,3} -5",
        "Plane: n={0,0.97014,-0.24254}, d=-0.72761",
    };

    EXPECT_EQ(expected_node, result.node);
    EXPECT_VEC_EQ(expected_surfaces, result.surfaces);
    EXPECT_FALSE(result.interior) << result.interior;
    EXPECT_VEC_SOFT_EQ((Real3{-2, -1, -1}), result.exterior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{2, 1, 1}), result.exterior.upper());

    GTEST_SKIP() << "twisty point sampling fails!";
    this->check_corners(result.node_id, pri, 0.1);
}

TEST_F(GenTrapTest, trap_even_twist)
{
    auto trap = GenTrap::from_trap(
        1, Turn{0}, Turn{0}, {1, 2, 2, Turn{0}}, {1, 2, 2, Turn{0.125}});

    static real_type const expected_twist_angles[] = {0.125, 0, 0.125, 0};
    EXPECT_VEC_SOFT_EQ(expected_twist_angles, this->get_twist_angles(trap));

    static real_type const expected_lower[] = {2, -1, 2, 1, -2, 1, -2, -1};
    static real_type const expected_upper[]
        = {0.5, -0.5, 1.5, 0.5, -0.5, 0.5, -1.5, -0.5};
    EXPECT_VEC_SOFT_EQ(expected_lower, to_vec(trap.lower()));
    EXPECT_VEC_SOFT_EQ(expected_upper, to_vec(trap.upper()));

    cout << repr(trap.lower()) << ',' << repr(trap.upper());
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
        // TODO: the scaled GQ should be normalized
        auto result = this->test("scaled",
                                 GenPrism(1,
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
        "GQuadric: {0,0,0} {0,1,0} {4,1,0} 0",
        "Plane: y=1",
        "Plane: x=-1",
        "Plane: x=1",
        "Plane: y=-2",
        "Plane: x=2",
        "Plane: y=2",
        "GQuadric: {0,0,0} {0,2,0} {8,2,0} 0",
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
        "left@t1,right@t3",
        "",
        "left@p2,right@p2",
        "",
        "left@p3",
        "",
        "right@p1",
        "",
        "",
        "scaled@p0",
        "scaled@p1",
        "",
        "scaled@p2",
        "",
        "scaled@t3",
        "",
    };
    EXPECT_VEC_EQ(expected_node_strings, node_strings);
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
               "Plane: n={0.70711,0.70711,0}, d=0"};

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
            = {"Plane: y=0", "Plane: n={0.70711,-0.70711,0}, d=0"};

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
    static char const* const expected_surfaces[] = {
        "Plane: z=-1.2",
        "Plane: z=1.2",
        "Plane: n={0.86603,0.5,0}, d=1",
        "Plane: n={0.86603,-0.5,0}, d=-1",
        "Plane: y=-1",
    };

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
    static char const* const expected_surfaces[] = {
        "Plane: z=-1.2",
        "Plane: z=1.2",
        "Plane: y=1",
        "Plane: n={0.86603,0.5,0}, d=-1",
        "Plane: n={0.86603,-0.5,0}, d=1",
    };

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
