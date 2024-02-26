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
    auto node_id
        = unit_builder_.insert_csg(Joined{op_and, std::move(css.nodes)});

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
}  // namespace test
}  // namespace orangeinp
}  // namespace celeritas
