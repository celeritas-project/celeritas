//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/CsgUnitBuilder.test.cc
//---------------------------------------------------------------------------//
#include "orange/orangeinp/detail/CsgUnitBuilder.hh"

#include "orange/orangeinp/CsgTestUtils.hh"
#include "orange/surf/Sphere.hh"
#include "orange/surf/SphereCentered.hh"

#include "celeritas_test.hh"

using N = celeritas::orangeinp::NodeId;
using V = celeritas::LocalVolumeId;

using namespace celeritas::orangeinp::test;

namespace celeritas
{
namespace orangeinp
{
namespace detail
{
namespace test
{
//---------------------------------------------------------------------------//

class CsgUnitBuilderTest : public ::celeritas::test::Test
{
  protected:
    using Tol = CsgUnitBuilder::Tol;

  protected:
    Tol tol = Tolerance<>::from_relative(1e-4);
};

TEST_F(CsgUnitBuilderTest, infinite)
{
    CsgUnit u;
    EXPECT_FALSE(u);

    CsgUnitBuilder builder(&u, tol);
    EXPECT_REAL_EQ(1e-4, builder.tol().rel);

    // Add a new 'true' node
    auto true_nid = builder.insert_csg(True{}).first;
    EXPECT_EQ(CsgTree::true_node_id(), true_nid);
    builder.insert_md(true_nid, "true");
    builder.set_bounds(true_nid, BoundingZone::from_infinite());

    // Add a volume and fill it with a material
    auto vid = builder.insert_volume(true_nid);
    EXPECT_EQ(V{0}, vid);
    builder.fill_volume(vid, MaterialId{123});

    EXPECT_TRUE(u);
    static char const* const expected_md_strings[] = {"true", ""};
    static char const* const expected_bound_strings[] = {"0: {inf, inf}"};
    static int const expected_volume_nodes[] = {0};
    static char const* const expected_fill_strings[] = {"m123"};
    static char const expected_tree_string[] = R"json(["t",["~",0]])json";

    EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
    EXPECT_VEC_EQ(expected_bound_strings, bound_strings(u));
    EXPECT_VEC_EQ(expected_volume_nodes, volume_nodes(u));
    EXPECT_VEC_EQ(expected_fill_strings, fill_strings(u));
    if (CELERITAS_USE_JSON)
    {
        EXPECT_JSON_EQ(expected_tree_string, tree_string(u));
    }
    EXPECT_EQ(NodeId{}, u.exterior);
}

TEST_F(CsgUnitBuilderTest, single_surface)
{
    CsgUnit u;
    CsgUnitBuilder builder(&u, tol);
    if (CELERITAS_DEBUG)
    {
        EXPECT_THROW(builder.surface<SphereCentered>(N{10}),
                     celeritas::DebugError);
    }

    // Add a surface and the corresponding node
    auto outside_nid = builder.insert_surface(SphereCentered{1.0}).first;
    EXPECT_EQ(N{2}, outside_nid);
    builder.insert_md(outside_nid, {"sphere", "o"});

    // Test accessing the constructed surface
    EXPECT_SOFT_EQ(1.0,
                   builder.surface<SphereCentered>(outside_nid).radius_sq());
    if (CELERITAS_DEBUG)
    {
        EXPECT_THROW(builder.surface<Sphere>(outside_nid),
                     celeritas::DebugError);
    }

    // Add a new 'inside sphere' node
    auto inside_nid = builder.insert_csg(Negated{outside_nid}).first;
    EXPECT_EQ(N{3}, inside_nid);
    builder.insert_md(inside_nid, {"sphere", "i"});
    builder.insert_md(inside_nid, {"sphere"});

    if (CELERITAS_DEBUG)
    {
        EXPECT_THROW(builder.surface<Sphere>(inside_nid),
                     celeritas::DebugError);
    }

    // Add a volume and fill it with a material
    auto inside_vid = builder.insert_volume(inside_nid);
    EXPECT_EQ(V{0}, inside_vid);
    builder.fill_volume(inside_vid, MaterialId{1});
    builder.insert_md(inside_nid, {"sphere_vol"});

    auto outside_vid = builder.insert_volume(outside_nid);
    EXPECT_EQ(V{1}, outside_vid);
    builder.fill_volume(outside_vid, MaterialId{0});
    builder.insert_md(outside_nid, {"exterior"});
    builder.set_exterior(outside_nid);

    EXPECT_TRUE(u);
    static char const* const expected_surface_strings[] = {"Sphere: r=1"};
    static char const* const expected_md_strings[]
        = {"", "", "exterior,sphere@o", "sphere,sphere@i,sphere_vol"};
    static int const expected_volume_nodes[] = {3, 2};
    static char const* const expected_fill_strings[] = {"m1", "m0"};
    static char const expected_tree_string[]
        = R"json(["t",["~",0],["S",0],["~",2]])json";

    EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
    EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
    EXPECT_VEC_EQ(expected_volume_nodes, volume_nodes(u));
    EXPECT_VEC_EQ(expected_fill_strings, fill_strings(u));
    if (CELERITAS_USE_JSON)
    {
        EXPECT_JSON_EQ(expected_tree_string, tree_string(u));
    }
    EXPECT_EQ(NodeId{2}, u.exterior);
}

TEST_F(CsgUnitBuilderTest, multi_level)
{
    CsgUnit u;
    CsgUnitBuilder builder(&u, tol);

    // Add inner surface
    auto [outer, ins] = builder.insert_surface(SphereCentered{1.0});
    EXPECT_TRUE(ins);
    EXPECT_EQ(N{2}, outer);
    builder.insert_md(outer, {"coating", "o"});

    // Add outer surface
    auto inner = builder.insert_surface(SphereCentered{0.75}).first;
    EXPECT_EQ(N{3}, inner);
    builder.insert_md(inner, {"coating", "i"});

    // Add "solid"
    auto neg_outer = builder.insert_csg(Negated{outer}).first;
    EXPECT_EQ(N{4}, neg_outer);
    auto coating = builder.insert_csg(Joined{op_and, {inner, neg_outer}}).first;
    EXPECT_EQ(N{5}, coating);
    builder.insert_md(coating, {"coating"});

    // Add coating volumes
    auto coating_vid = builder.insert_volume(coating);
    EXPECT_EQ(V{0}, coating_vid);
    builder.insert_md(coating, {"coating_vol"});
    builder.fill_volume(coating_vid, MaterialId{1});

    // Add inner universe
    {
        // Pretend user has a small gap within the tolerance
        auto ins = builder.insert_surface(SphereCentered{0.75 - 1e-5});
        EXPECT_EQ(inner, ins.first);
        EXPECT_FALSE(ins.second);
    }
    auto neg_inner = builder.insert_csg(Negated{inner}).first;
    EXPECT_EQ(N{6}, neg_inner);
    auto inner_vid = builder.insert_volume(neg_inner);
    EXPECT_EQ(V{1}, inner_vid);
    builder.insert_md(neg_inner, {"daughter"});
    builder.fill_volume(inner_vid, UniverseId{1}, Translation{{1, 2, 3}});

    // Add outside
    auto outside_vid = builder.insert_volume(outer);
    EXPECT_EQ(V{2}, outside_vid);
    builder.fill_volume(outside_vid, MaterialId{0});
    builder.insert_md(outer, {"exterior"});
    builder.set_exterior(outer);

    EXPECT_TRUE(u);
    static char const* const expected_surface_strings[]
        = {"Sphere: r=1", "Sphere: r=0.75", "Sphere: r=0.74999"};
    static char const* const expected_md_strings[] = {"",
                                                      "",
                                                      "coating@o,exterior",
                                                      "coating@i",
                                                      "",
                                                      "coating,coating_vol",
                                                      "daughter"};
    static int const expected_volume_nodes[] = {5, 6, 2};
    static char const* const expected_fill_strings[]
        = {"m1", "{1,{{1,2,3}}", "m0"};
    static char const expected_tree_string[]
        = R"json(["t",["~",0],["S",0],["S",1],["~",2],["&",[3,4]],["~",3]])json";

    EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
    EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
    EXPECT_VEC_EQ(expected_volume_nodes, volume_nodes(u));
    EXPECT_VEC_EQ(expected_fill_strings, fill_strings(u));
    if (CELERITAS_USE_JSON)
    {
        EXPECT_JSON_EQ(expected_tree_string, tree_string(u));
    }
    EXPECT_EQ(NodeId{2}, u.exterior);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
