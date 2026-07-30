//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/IntersectSurfaceBuilder.test.cc
//---------------------------------------------------------------------------//
#include "orange/orangeinp/IntersectSurfaceBuilder.hh"

#include "orange/MatrixUtils.hh"
#include "orange/orangeinp/detail/IntersectSurfaceState.hh"

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

class IntersectSurfaceBuilderTest : public ObjectTestBase
{
  protected:
    using State = detail::IntersectSurfaceState;

    Tol tolerance() const override { return Tol::from_relative(1e-4); }

    State make_state() const
    {
        State result;
        result.transform = &transform_;
        result.make_face_name = FaceNamer{"c."};
        return result;
    }

  protected:
    VariantTransform transform_;
};

void x_slab(IntersectSurfaceBuilder& build)
{
    build(Sense::outside, PlaneX{-0.5});
    build(Sense::inside, PlaneX{0.5});
}

TEST_F(IntersectSurfaceBuilderTest, no_transform)
{
    auto inf = std::numeric_limits<real_type>::infinity();
    {
        SCOPED_TRACE("z_hemi");
        auto css = this->make_state();
        css.object_name = "zh";
        IntersectSurfaceBuilder build{&this->unit_builder(), &css};
        build(PlaneZ{0.0});
        build(SphereCentered{1.0});

        // clang-format off
        static real_type const expected_local_bz[] = {-0.86602540378444,
            -0.86602540378444, -0.86602540378444, 0.86602540378444,
            0.86602540378444, 0, -1, -1, -1, 1, 1, 0, 1};
        static int const expected_nodes[] = {3, 5};
        // clang-format on
        EXPECT_VEC_SOFT_EQ(expected_local_bz, flattened(css.local_bzone));
        EXPECT_VEC_SOFT_EQ(expected_local_bz, flattened(css.global_bzone));
        EXPECT_VEC_EQ(expected_nodes, to_vec_int(css.nodes));
    }
    {
        SCOPED_TRACE("reverse_hemi");
        auto css = this->make_state();
        css.object_name = "rh";
        IntersectSurfaceBuilder build{&this->unit_builder(), &css};
        build(Sense::outside, PlaneZ{1e-5});
        build(Sense::inside, SphereCentered{1.0});

        // clang-format off
        static real_type const expected_local_bz[] = {
            -0.86602540378444, -0.86602540378444, 0,
            0.86602540378444, 0.86602540378444, 0.86602540378444,
            -1, -1, 0,
            1, 1, 1,
            1};
        static int const expected_nodes[] = {2, 5};
        // clang-format on
        EXPECT_VEC_SOFT_EQ(expected_local_bz, flattened(css.local_bzone));
        EXPECT_VEC_SOFT_EQ(expected_local_bz, flattened(css.global_bzone));
        EXPECT_VEC_EQ(expected_nodes, to_vec_int(css.nodes));
    }
    {
        SCOPED_TRACE("dedupe hemi");
        auto css = this->make_state();
        css.object_name = "dh";
        IntersectSurfaceBuilder build{&this->unit_builder(), &css};
        build(PlaneZ{1e-5});
        build(SphereCentered{1.0});

        // clang-format off
        static real_type const expected_local_bz[] = {-0.86602540378444,
            -0.86602540378444, -0.86602540378444, 0.86602540378444,
            0.86602540378444, 0, -1, -1, -1, 1, 1, 0, 1};
        static int const expected_nodes[] = {3, 5};
        // clang-format on
        EXPECT_VEC_SOFT_EQ(expected_local_bz, flattened(css.local_bzone));
        EXPECT_VEC_SOFT_EQ(expected_local_bz, flattened(css.global_bzone));
        EXPECT_VEC_EQ(expected_nodes, to_vec_int(css.nodes));
    }
    {
        SCOPED_TRACE("slab");
        auto css = this->make_state();
        css.object_name = "sl";
        IntersectSurfaceBuilder build{&this->unit_builder(), &css};
        x_slab(build);

        static real_type const expected_local_bz[] = {
            -0.5, -inf, -inf, 0.5, inf, inf, -0.5, -inf, -inf, 0.5, inf, inf, 1};
        static real_type const expected_global_bz[] = {
            -0.5, -inf, -inf, 0.5, inf, inf, -0.5, -inf, -inf, 0.5, inf, inf, 1};
        static int const expected_nodes[] = {6, 8};
        EXPECT_VEC_SOFT_EQ(expected_local_bz, flattened(css.local_bzone));
        EXPECT_VEC_SOFT_EQ(expected_global_bz, flattened(css.global_bzone));
        EXPECT_VEC_EQ(expected_nodes, to_vec_int(css.nodes));
    }

    // clang-format off
    static char const* const expected_surface_strings[]
        = {"Plane: z=0", "Sphere: r=1", "Plane: x=-0.5", "Plane: x=0.5"};
    static char const* const expected_md_strings[] = {"", "",
        "dh@c.pz,rh@c.mz,zh@c.pz", "", "dh@c.s,rh@c.s,zh@c.s", "", "sl@c.mx",
        "sl@c.px", ""};
    static char const expected_tree_string[]
        = R"json(["t",["~",0],["S",0],["~",2],["S",1],["~",4],["S",2],["S",3],["~",7]])json";
    // clang-format on

    auto const& u = this->unit();
    EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
    EXPECT_JSON_EQ(expected_tree_string, tree_string(u));
    EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
}

TEST_F(IntersectSurfaceBuilderTest, translate)
{
    auto inf = std::numeric_limits<real_type>::infinity();
    transform_ = Translation{Real3{1, 2, 3}};
    {
        SCOPED_TRACE("slab");
        auto css = this->make_state();
        css.object_name = "sl";
        IntersectSurfaceBuilder build{&this->unit_builder(), &css};
        x_slab(build);

        // clang-format off
        static real_type const expected_local_bz[] = {-0.5, -inf, -inf, 0.5,
            inf, inf, -0.5, -inf, -inf, 0.5, inf, inf, 1};
        static real_type const expected_global_bz[] = {0.5, -inf, -inf, 1.5,
            inf, inf, 0.5, -inf, -inf, 1.5, inf, inf, 1};
        static int const expected_nodes[] = {2, 4};
        // clang-format on

        EXPECT_VEC_SOFT_EQ(expected_local_bz, flattened(css.local_bzone));
        EXPECT_VEC_SOFT_EQ(expected_global_bz, flattened(css.global_bzone));
        EXPECT_VEC_EQ(expected_nodes, to_vec_int(css.nodes));
    }
    {
        SCOPED_TRACE("sphere");
        auto css = this->make_state();
        css.object_name = "sph";
        IntersectSurfaceBuilder build{&this->unit_builder(), &css};
        build(SphereCentered{1.0});

        // clang-format off
        static real_type const expected_local_bz[] = {-0.86602540378444,
            -0.86602540378444, -0.86602540378444, 0.86602540378444,
            0.86602540378444, 0.86602540378444, -1, -1, -1, 1, 1, 1, 1};
        static real_type const expected_global_bz[] = {0.13397459621556,
            1.1339745962156, 2.1339745962156, 1.8660254037844, 2.8660254037844,
            3.8660254037844, 0, 1, 2, 2, 3, 4, 1};
        static int const expected_nodes[] = {6};
        // clang-format on

        EXPECT_VEC_SOFT_EQ(expected_local_bz, flattened(css.local_bzone));
        EXPECT_VEC_SOFT_EQ(expected_global_bz, flattened(css.global_bzone));
        EXPECT_VEC_EQ(expected_nodes, to_vec_int(css.nodes));
    }
    transform_ = Translation{Real3{2, 0, 0}};
    {
        SCOPED_TRACE("slab shared");
        auto css = this->make_state();
        css.object_name = "ss";
        IntersectSurfaceBuilder build{&this->unit_builder(), &css};
        x_slab(build);

        // clang-format off
        static real_type const expected_local_bz[] = {-0.5, -inf, -inf, 0.5,
            inf, inf, -0.5, -inf, -inf, 0.5, inf, inf, 1};
        static real_type const expected_global_bz[] = {1.5, -inf, -inf, 2.5,
            inf, inf, 1.5, -inf, -inf, 2.5, inf, inf, 1};
        static int const expected_nodes[] = {3, 8};
        // clang-format on

        EXPECT_VEC_SOFT_EQ(expected_local_bz, flattened(css.local_bzone));
        EXPECT_VEC_SOFT_EQ(expected_global_bz, flattened(css.global_bzone));
        EXPECT_VEC_EQ(expected_nodes, to_vec_int(css.nodes));
    }

    static char const* const expected_surface_strings[] = {
        "Plane: x=0.5",
        "Plane: x=1.5",
        "Sphere: r=1 at {1,2,3}",
        "Plane: x=2.5",
    };
    static char const* const expected_md_strings[] = {
        "", "", "sl@c.mx", "sl@c.px,ss@c.mx", "", "sph@c.s", "", "ss@c.px", ""};
    static char const expected_tree_string[]
        = R"json(["t",["~",0],["S",0],["S",1],["~",3],["S",2],["~",5],["S",3],["~",7]])json";

    auto const& u = this->unit();
    EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
    EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
    EXPECT_JSON_EQ(expected_tree_string, tree_string(u));
}

TEST_F(IntersectSurfaceBuilderTest, transform)
{
    transform_
        = Transformation{make_rotation(Axis::x, Turn{0.25}), Real3{0, 0, 1}};
    {
        SCOPED_TRACE("hemi");
        auto css = this->make_state();
        css.object_name = "h";
        IntersectSurfaceBuilder build{&this->unit_builder(), &css};
        build(PlaneZ{1e-5});
        build(SphereCentered{1.0});

        // clang-format off
        static real_type const expected_local_bz[] = {-0.86602540378444,
            -0.86602540378444, -0.86602540378444, 0.86602540378444,
            0.86602540378444, 0, -1, -1, -1, 1, 1, 0, 1};
        static real_type const expected_global_bz[] = {-0.86602540378444, 0,
            0.13397459621556, 0.86602540378444, 0.86602540378444,
            1.8660254037844, -1, 0, 0, 1, 1, 2, 1};
        static int const expected_nodes[] = {2, 4};

        // clang-format on
        EXPECT_VEC_SOFT_EQ(expected_local_bz, flattened(css.local_bzone));
        EXPECT_VEC_SOFT_EQ(expected_global_bz, flattened(css.global_bzone));
        EXPECT_VEC_EQ(expected_nodes, to_vec_int(css.nodes));
    }

    static char const* const expected_surface_strings[]
        = {"Plane: y=0", "Sphere: r=1 at {0,0,1}"};
    static char const expected_tree_string[]
        = R"json(["t",["~",0],["S",0],["S",1],["~",3]])json";
    static char const* const expected_md_strings[]
        = {"", "", "h@c.pz", "h@c.s", ""};

    auto const& u = this->unit();
    EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
    EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
    EXPECT_JSON_EQ(expected_tree_string, tree_string(u));
}

TEST_F(IntersectSurfaceBuilderTest, finite_extents)
{
    auto inf = std::numeric_limits<real_type>::infinity();
    this->reset(BBox{{-10, -10, -10}, {10, 10, 10}});
    {
        SCOPED_TRACE("slab");
        auto css = this->make_state();
        css.object_name = "sl";
        IntersectSurfaceBuilder build{&this->unit_builder(), &css};
        x_slab(build);

        static real_type const expected_local_bz[] = {
            -0.5, -inf, -inf, 0.5, inf, inf, -0.5, -inf, -inf, 0.5, inf, inf, 1};
        static real_type const expected_global_bz[]
            = {-0.5, -10, -10, 0.5, 10, 10, -0.5, -10, -10, 0.5, 10, 10, 1};
        static int const expected_nodes[] = {2, 4};
        EXPECT_VEC_SOFT_EQ(expected_local_bz, flattened(css.local_bzone));
        EXPECT_VEC_SOFT_EQ(expected_global_bz, flattened(css.global_bzone));
        EXPECT_VEC_EQ(expected_nodes, to_vec_int(css.nodes));
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace orangeinp
}  // namespace celeritas
