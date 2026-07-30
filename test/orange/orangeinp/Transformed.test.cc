//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/Transformed.test.cc
//---------------------------------------------------------------------------//
#include "orange/orangeinp/Transformed.hh"

#include "corecel/ScopedLogStorer.hh"
#include "corecel/io/Logger.hh"
#include "orange/MatrixUtils.hh"
#include "orange/orangeinp/Shape.hh"

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

class TransformedTest : public ObjectTestBase
{
  protected:
    Tol tolerance() const override { return Tol::from_relative(1e-4); }
};

//---------------------------------------------------------------------------//
TEST_F(TransformedTest, notran)
{
    auto sphshape = std::make_shared<SphereShape>("sph", Sphere{1.0});
    this->build_volume(*Transformed::or_object(sphshape, NoTransformation{}));

    static char const* const expected_surface_strings[] = {"Sphere: r=1"};
    static char const* const expected_trans_strings[] = {"3: t=0 -> {}"};
    static int const expected_volume_nodes[] = {3};
    static char const expected_tree_string[]
        = R"json(["t",["~",0],["S",0],["~",2]])json";

    auto const& u = this->unit();
    EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
    EXPECT_VEC_EQ(expected_trans_strings, transform_strings(u));
    EXPECT_VEC_EQ(expected_volume_nodes, volume_nodes(u));
    EXPECT_JSON_EQ(expected_tree_string, tree_string(u));
}

TEST_F(TransformedTest, single)
{
    auto sphshape = std::make_shared<SphereShape>("sph", Sphere{1.0});
    this->build_volume(Transformed{sphshape, Translation{{0, 0, 1.0}}});

    static char const* const expected_surface_strings[]
        = {"Sphere: r=1 at {0,0,1}"};
    static char const* const expected_volume_strings[] = {"-0"};
    static char const* const expected_md_strings[] = {"", "", "sph@s", "sph"};

    auto const& u = this->unit();
    EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
    EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
    EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
}

TEST_F(TransformedTest, several)
{
    auto sphshape = std::make_shared<SphereShape>("sph", Sphere{2.0});
    auto cylshape = std::make_shared<CylinderShape>("cyl", Cylinder{1.0, 2.0});

    // Build original and translated spheres
    this->build_volume(*sphshape);
    this->build_volume(*cylshape);

    this->build_volume(Transformed{sphshape, Translation{{3, 0, 0}}});
    this->build_volume(Transformed{cylshape, Translation{{0, 0, 4}}});

    this->build_volume(Transformed{
        cylshape,
        Transformation{make_rotation(Axis::x, Turn{0.25}), {0, 5, 0}}});

    static char const* const expected_surface_strings[] = {
        "Sphere: r=2",
        "Plane: z=-2",
        "Plane: z=2",
        "Cyl z: r=1",
        "Sphere: r=2 at {3,0,0}",
        "Plane: z=6",
        "Plane: y=7",
        "Plane: y=3",
        "Cyl y: r=1",
    };
    static char const* const expected_volume_strings[] = {
        "-0",
        "all(+1, -2, -3)",
        "-4",
        "all(+2, -3, -5)",
        "all(-6, +7, -8)",
    };
    static char const* const expected_md_strings[]
        = {"",       "",       "sph@s", "sph",    "cyl@mz", "cyl@mz,cyl@pz",
           "",       "cyl@cz", "",      "cyl",    "sph@s",  "sph",
           "cyl@pz", "",       "cyl",   "cyl@mz", "",       "cyl@pz",
           "cyl@cz", "",       "cyl"};
    static char const* const expected_bound_strings[] = {
        "3: {{{-1.73,-1.73,-1.73}, {1.73,1.73,1.73}}, {{-2,-2,-2}, {2,2,2}}}",
        "9: {{{-0.707,-0.707,-2}, {0.707,0.707,2}}, {{-1,-1,-2}, {1,1,2}}}",
        "11: {{{1.27,-1.73,-1.73}, {4.73,1.73,1.73}}, {{1,-2,-2}, {5,2,2}}}",
        "14: {{{-0.707,-0.707,2}, {0.707,0.707,6}}, {{-1,-1,2}, {1,1,6}}}",
        "20: {{{-0.707,3,-0.707}, {0.707,7,0.707}}, {{-1,3,-1}, {1,7,1}}}"};
    static char const* const expected_trans_strings[] = {
        "3: t=0 -> {}",
        "9: t=0",
        "11: t=1 -> {{3,0,0}}",
        "14: t=2 -> {{0,0,4}}",
        "20: t=3 -> {{{1,0,0},{0,0,-1},{0,1,0}}, {0,5,0}}",
    };

    auto const& u = this->unit();
    EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
    EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
    EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
    EXPECT_VEC_EQ(expected_bound_strings, bound_strings(u));
    EXPECT_VEC_EQ(expected_trans_strings, transform_strings(u));
}

TEST_F(TransformedTest, stacked)
{
    auto sph = std::make_shared<SphereShape>("sph", Sphere{1.0});
    auto trsph = std::make_shared<Transformed>(sph, Translation{{2, 0, 0}});
    auto tr2sph = std::make_shared<Transformed>(
        trsph, Transformation{make_rotation(Axis::z, Turn{0.25}), {1, 0, 0}});
    auto tr3sph = std::make_shared<Transformed>(tr2sph, Translation{{0, 0, 3}});

    this->build_volume(*tr3sph);
    this->build_volume(*tr2sph);
    this->build_volume(*trsph);
    this->build_volume(*sph);

    static char const* const expected_surface_strings[] = {
        "Sphere: r=1 at {1,2,3}",
        "Sphere: r=1 at {1,2,0}",
        "Sphere: r=1 at {2,0,0}",
        "Sphere: r=1",
    };
    static char const* const expected_volume_strings[]
        = {"-0", "-1", "-2", "-3"};
    static char const* const expected_md_strings[] = {
        "", "", "sph@s", "sph", "sph@s", "sph", "sph@s", "sph", "sph@s", "sph"};
    static char const* const expected_trans_strings[] = {
        "3: t=3 -> {{{0,-1,0},{1,0,0},{0,0,1}}, {1,2,3}}",
        "5: t=5 -> {{{0,-1,0},{1,0,0},{0,0,1}}, {1,2,0}}",
        "7: t=6 -> {{2,0,0}}",
        "9: t=0 -> {}",
    };

    auto const& u = this->unit();
    EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
    EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
    EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
    EXPECT_VEC_EQ(expected_trans_strings, transform_strings(u));
}

TEST_F(TransformedTest, inverse)
{
    auto cylshape = std::make_shared<CylinderShape>("cyl", Cylinder{1.0, 1.0});
    // Transformed: top face is at y=3, bottom at y=1
    auto transformed = std::make_shared<Transformed>(
        cylshape,
        Transformation{make_rotation(Axis::x, Turn{-0.25}), {0, 2, 0}});

    // Build original, transformed, and anti-transformed
    this->build_volume(*cylshape);
    this->build_volume(*transformed);
    {
        celeritas::test::ScopedLogStorer scoped_log_{
            &celeritas::world_logger()};
        this->build_volume(Transformed{
            transformed,
            Transformation{make_rotation(Axis::x, Turn{0.25}), {0, 0, -2}}});
        EXPECT_TRUE(scoped_log_.empty()) << scoped_log_;
    }

    static char const* const expected_surface_strings[] = {
        "Plane: z=-1",
        "Plane: z=1",
        "Cyl z: r=1",
        "Plane: y=1",
        "Plane: y=3",
        "Cyl y: r=1",
    };
    static char const* const expected_volume_strings[]
        = {"all(+0, -1, -2)", "all(+3, -4, -5)", "all(+0, -1, -2)"};
    static char const* const expected_md_strings[] = {
        "",
        "",
        "cyl@mz",
        "cyl@pz",
        "",
        "cyl@cz",
        "",
        "cyl",
        "cyl@mz",
        "cyl@pz",
        "",
        "cyl@cz",
        "",
        "cyl",
    };
    static char const* const expected_trans_strings[]
        = {"7: t=0 -> {}", "13: t=1 -> {{{1,0,0},{0,0,1},{0,-1,0}}, {0,2,0}}"};

    auto const& u = this->unit();
    EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
    EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
    EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
    EXPECT_VEC_EQ(expected_trans_strings, transform_strings(u));
}

TEST_F(TransformedTest, deduplicated_inverse)
{
    Transformation const tr{make_rotation(Axis::x, Turn{-0.25}), {0, 2, 0}};
    Transformation const inv_tr{make_rotation(Axis::x, Turn{0.25}), {0, 0, -2}};

    auto sph_inner = std::make_shared<SphereShape>("sphi", Sphere{1.0});
    auto sph_outer = std::make_shared<SphereShape>("spho", Sphere{2.0});
    auto tr_inner = std::make_shared<Transformed>(sph_inner, tr);

    // Build outer and anti-transformed
    this->build_volume(*sph_outer);
    this->build_volume(Transformed{tr_inner, inv_tr});

    static char const* const expected_surface_strings[]
        = {"Sphere: r=2", "Sphere: r=1"};
    static char const* const expected_volume_strings[] = {"-0", "-1"};
    static char const* const expected_trans_strings[]
        = {"3: t=0 -> {}", "5: t=0"};
    static int const expected_volume_nodes[] = {3, 5};

    auto const& u = this->unit();
    EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
    EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
    EXPECT_VEC_EQ(expected_trans_strings, transform_strings(u));
    EXPECT_VEC_EQ(expected_volume_nodes, volume_nodes(u));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace orangeinp
}  // namespace celeritas
