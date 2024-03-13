//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/UnitProto.test.cc
//---------------------------------------------------------------------------//
#include "orange/orangeinp/UnitProto.hh"

#include <memory>

#include "corecel/io/Join.hh"
#include "orange/orangeinp/CsgObject.hh"
#include "orange/orangeinp/Shape.hh"
#include "orange/orangeinp/Transformed.hh"
#include "orange/orangeinp/detail/CsgUnit.hh"

#include "CsgTestUtils.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace orangeinp
{
namespace test
{
//---------------------------------------------------------------------------//
// Type aliases
//---------------------------------------------------------------------------//
using SPConstObject = std::shared_ptr<ObjectInterface const>;
using SPConstProto = std::shared_ptr<ProtoInterface const>;
inline constexpr auto is_global = UnitProto::ExteriorBoundary::is_global;
inline constexpr auto is_daughter = UnitProto::ExteriorBoundary::is_daughter;

//---------------------------------------------------------------------------//
// Construction helper functions
//---------------------------------------------------------------------------//
SPConstObject make_sph(std::string&& label, real_type radius)
{
    return std::make_shared<SphereShape>(std::move(label), Sphere{radius});
}

SPConstObject
make_cyl(std::string&& label, real_type radius, real_type halfheight)
{
    return std::make_shared<CylinderShape>(std::move(label),
                                           Cylinder{radius, halfheight});
}

SPConstObject make_translated(SPConstObject&& obj, Real3 const& trans)
{
    return std::make_shared<Transformed>(std::move(obj), Translation{trans});
}

SPConstProto make_daughter(std::string label)
{
    UnitProto::Input inp;
    inp.boundary.interior = make_sph(label + ":ext", 1);
    inp.fill = MaterialId{0};
    inp.label = std::move(label);

    return std::make_shared<UnitProto>(std::move(inp));
}

std::string proto_labels(ProtoInterface::VecProto const& vp)
{
    return to_string(join_stream(
        vp.begin(), vp.end(), ",", [](std::ostream& os, ProtoInterface const* p) {
            if (!p)
            {
                os << "<null>";
            }
            else
            {
                os << p->label();
            }
        }));
}

//---------------------------------------------------------------------------//
class UnitProtoTest : public ::celeritas::test::Test
{
  protected:
    using Unit = detail::CsgUnit;
    using Tol = Tolerance<>;

    Tolerance<> tol_ = Tol::from_relative(1e-5);
};

//---------------------------------------------------------------------------//
using LeafTest = UnitProtoTest;

// All space is explicitly accounted for
TEST_F(LeafTest, explicit_exterior)
{
    UnitProto::Input inp;
    inp.boundary.interior = make_cyl("bound", 1.0, 1.0);
    inp.boundary.zorder = ZOrder::media;
    inp.label = "leaf";
    inp.materials.push_back(
        {make_translated(make_cyl("bottom", 1, 0.5), {0, 0, -0.5}),
         MaterialId{1}});
    inp.materials.push_back(
        {make_translated(make_cyl("top", 1, 0.5), {0, 0, 0.5}), MaterialId{2}});
    UnitProto const proto{std::move(inp)};

    EXPECT_EQ("", proto_labels(proto.daughters()));

    {
        auto u = proto.build(tol_, is_global);

        static char const* const expected_surface_strings[]
            = {"Plane: z=-1", "Plane: z=1", "Cyl z: r=1", "Plane: z=0"};
        static char const* const expected_volume_strings[]
            = {"!all(+0, -1, -2)", "all(+0, -2, -3)", "all(-1, -2, +3)"};
        static char const* const expected_md_strings[] = {
            "",
            "",
            "bottom@mz,bound@mz",
            "bound@pz,top@pz",
            "",
            "bottom@cz,bound@cz,top@cz",
            "",
            "bound",
            "[EXTERIOR]",
            "bottom@pz,top@mz",
            "",
            "bottom",
            "top",
        };
        static char const* const expected_fill_strings[]
            = {"<UNASSIGNED>", "m1", "m2"};

        EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
        EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
        EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
        EXPECT_VEC_EQ(expected_fill_strings, fill_strings(u));
        EXPECT_EQ(MaterialId{}, u.background);
    }
    {
        auto u = proto.build(tol_, is_daughter);
        static char const* const expected_volume_strings[] = {"F", "-3", "+3"};

        EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
    }
}

// Inside of the "mother" volume is implicit
TEST_F(LeafTest, implicit_exterior)
{
    UnitProto::Input inp;
    inp.boundary.interior = make_cyl("bound", 1.0, 1.0);
    inp.boundary.zorder = ZOrder::exterior;
    inp.fill = MaterialId{0};
    inp.label = "leaf";
    inp.materials.push_back({make_cyl("middle", 1, 0.5), MaterialId{1}});
    UnitProto const proto{std::move(inp)};

    {
        auto u = proto.build(tol_, is_global);

        static char const* const expected_surface_strings[] = {
            "Plane: z=-1",
            "Plane: z=1",
            "Cyl z: r=1",
            "Plane: z=-0.5",
            "Plane: z=0.5",
        };
        static char const* const expected_volume_strings[]
            = {"!all(+0, -1, -2)", "all(-2, +3, -4)"};
        static char const* const expected_fill_strings[]
            = {"<UNASSIGNED>", "m1"};

        EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
        EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
        EXPECT_VEC_EQ(expected_fill_strings, fill_strings(u));
        EXPECT_EQ(MaterialId{0}, u.background);
    }
    {
        auto u = proto.build(tol_, is_daughter);

        static char const* const expected_volume_strings[]
            = {"F", "all(+3, -4)"};
        EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
        EXPECT_EQ(MaterialId{0}, u.background);
    }
}

//---------------------------------------------------------------------------//
using MotherTest = UnitProtoTest;

TEST_F(MotherTest, explicit_exterior)
{
    UnitProto::Input inp;
    inp.boundary.interior = make_sph("bound", 10.0);
    inp.boundary.zorder = ZOrder::media;
    inp.label = "mother";
    inp.materials.push_back(
        {make_translated(make_sph("leaf", 1), {0, 0, -5}), MaterialId{1}});
    inp.materials.push_back(
        {make_translated(make_sph("leaf2", 1), {0, 0, 5}), MaterialId{2}});
    inp.daughters.push_back({make_daughter("d1"), Translation{{0, 5, 0}}});
    inp.daughters.push_back(
        {make_daughter("d2"),
         Transformation{make_rotation(Axis::x, Turn{0.25}), {0, -5, 0}}});

    // Construct "inside" cell
    std::vector<std::pair<Sense, SPConstObject>> interior
        = {{Sense::inside, inp.boundary.interior}};
    for (auto const& m : inp.materials)
    {
        interior.push_back({Sense::outside, m.interior});
    }
    for (auto const& d : inp.daughters)
    {
        interior.push_back({Sense::outside, d.make_interior()});
    }
    inp.materials.push_back(
        {make_rdv("interior", std::move(interior)), MaterialId{3}});

    UnitProto const proto{std::move(inp)};

    EXPECT_EQ("d1,d2", proto_labels(proto.daughters()));

    {
        auto u = proto.build(tol_, is_global);

        static char const* const expected_surface_strings[] = {
            "Sphere: r=10",
            "Sphere: r=1 at {0,5,0}",
            "Sphere: r=1 at {0,-5,0}",
            "Sphere: r=1 at {0,0,-5}",
            "Sphere: r=1 at {0,0,5}",
        };
        static char const* const expected_volume_strings[] = {
            "+0",
            "-1",
            "-2",
            "-3",
            "-4",
            "all(-0, +1, +2, +3, +4)",
        };
        static char const* const expected_md_strings[] = {"",
                                                          "",
                                                          "[EXTERIOR],bound@s",
                                                          "bound",
                                                          "d1:ext@s",
                                                          "d1:ext",
                                                          "d2:ext@s",
                                                          "d2:ext",
                                                          "leaf@s",
                                                          "leaf",
                                                          "leaf2@s",
                                                          "leaf2",
                                                          "interior"};
        static char const* const expected_trans_strings[] = {
            "2: t=0 -> {}",
            "3: t=0",
            "4: t=0",
            "5: t=1 -> {{0,5,0}}",
            "6: t=0",
            "7: t=2 -> {{{1,0,0},{0,0,-1},{0,1,0}}, {0,-5,0}}",
            "8: t=0",
            "9: t=3 -> {{0,0,-5}}",
            "10: t=0",
            "11: t=4 -> {{0,0,5}}",
            "12: t=0",
        };
        static char const* const expected_fill_strings[] = {
            "<UNASSIGNED>",
            "{u=0, t=1}",
            "{u=1, t=2}",
            "m1",
            "m2",
            "m3",
        };
        static int const expected_volume_nodes[] = {2, 5, 7, 9, 11, 12};

        EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
        EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
        EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
        EXPECT_VEC_EQ(expected_trans_strings, transform_strings(u));
        EXPECT_VEC_EQ(expected_fill_strings, fill_strings(u));
        EXPECT_VEC_EQ(expected_volume_nodes, volume_nodes(u));
        EXPECT_EQ(MaterialId{}, u.background);
    }
    {
        auto u = proto.build(tol_, is_daughter);
        static char const* const expected_volume_strings[]
            = {"F", "-1", "-2", "-3", "-4", "all(+1, +2, +3, +4)"};
        EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
    }
}

TEST_F(MotherTest, implicit_exterior)
{
    UnitProto::Input inp;
    inp.boundary.interior = make_sph("bound", 10.0);
    inp.boundary.zorder = ZOrder::media;
    inp.label = "mother";
    inp.materials.push_back(
        {make_translated(make_sph("leaf", 1), {0, 0, -5}), MaterialId{1}});
    inp.materials.push_back(
        {make_translated(make_sph("leaf2", 1), {0, 0, 5}), MaterialId{2}});
    inp.daughters.push_back({make_daughter("d1"), Translation{{0, 5, 0}}});
    inp.daughters.push_back(
        {make_daughter("d2"),
         Transformation{make_rotation(Axis::x, Turn{0.25}), {0, -5, 0}}});
    inp.fill = MaterialId{3};

    UnitProto const proto{std::move(inp)};

    EXPECT_EQ("d1,d2", proto_labels(proto.daughters()));

    {
        auto u = proto.build(tol_, is_global);
        static char const* const expected_volume_strings[]
            = {"+0", "-1", "-2", "-3", "-4"};
        static int const expected_volume_nodes[] = {2, 5, 7, 9, 11};

        EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
        EXPECT_VEC_EQ(expected_volume_nodes, volume_nodes(u));
        EXPECT_EQ(MaterialId{3}, u.background);
    }
    {
        auto u = proto.build(tol_, is_daughter);
        static char const* const expected_volume_strings[]
            = {"F", "-1", "-2", "-3", "-4"};
        EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
    }
}

TEST_F(MotherTest, fuzziness)
{
    UnitProto::Input inp;
    inp.boundary.interior = make_sph("bound", 10.0);
    inp.boundary.zorder = ZOrder::media;
    inp.label = "fuzzy";
    inp.daughters.push_back({make_daughter("d1"), {}});
    inp.materials.push_back(
        {make_rdv("interior",
                  {{Sense::inside, inp.boundary.interior},
                   {Sense::outside, make_sph("similar", 1.0001)}}),
         MaterialId{1}});

    UnitProto const proto{std::move(inp)};

    EXPECT_EQ("d1", proto_labels(proto.daughters()));

    {
        auto u = proto.build(tol_, is_global);
        static char const* const expected_surface_strings[]
            = {"Sphere: r=10", "Sphere: r=1", "Sphere: r=1.0001"};
        static char const* const expected_volume_strings[]
            = {"+0", "-1", "all(-0, +2)"};
        static char const* const expected_md_strings[] = {"",
                                                          "",
                                                          "[EXTERIOR],bound@s",
                                                          "bound",
                                                          "d1:ext@s",
                                                          "d1:ext",
                                                          "similar@s",
                                                          "similar",
                                                          "interior"};
        EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
        EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
        EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
    }
    {
        // Simplify with lower tolerance because the user has tried to avoid
        // overlap by adding .0001 to the "similar" shape
        auto u = proto.build(Tol::from_relative(1e-3), is_global);
        static char const* const expected_volume_strings[]
            = {"+0", "-1", "all(-0, +1)"};
        EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace orangeinp
}  // namespace celeritas
