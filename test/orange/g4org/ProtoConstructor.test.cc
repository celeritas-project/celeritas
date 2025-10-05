//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/g4org/ProtoConstructor.test.cc
//---------------------------------------------------------------------------//
#include "orange/g4org/ProtoConstructor.hh"

#include "corecel/StringSimplifier.hh"
#include "corecel/io/Repr.hh"
#include "geocel/GeantGeoParams.hh"
#include "orange/g4org/PhysicalVolumeConverter.hh"
#include "orange/orangeinp/CsgTestUtils.hh"
#include "orange/orangeinp/detail/CsgUnit.hh"
#include "orange/orangeinp/detail/ProtoMap.hh"

#include "GeantLoadTestBase.hh"
#include "celeritas_test.hh"

using namespace celeritas::orangeinp::test;
using celeritas::orangeinp::UnitProto;
using celeritas::orangeinp::detail::ProtoMap;
using celeritas::test::StringSimplifier;

namespace celeritas
{
namespace g4org
{
namespace test
{
//---------------------------------------------------------------------------//
class ProtoConstructorTest : public GeantLoadTestBase
{
  protected:
    using Unit = orangeinp::detail::CsgUnit;
    using Tol = Tolerance<>;

    std::shared_ptr<UnitProto> load(std::string const& basename)
    {
        // Load GDML into Geant4
        this->load_test_gdml(basename);

        // Convert volumes into ORANGE representation
        auto const& geant_geo = this->geo();
        PhysicalVolumeConverter make_pv(geant_geo, [] {
            PhysicalVolumeConverter::Options opts;
            opts.verbose = false;
            opts.scale = 0.1;
            return opts;
        }());
        PhysicalVolume world = make_pv(*geant_geo.world());

        EXPECT_TRUE(std::holds_alternative<NoTransformation>(world.transform));
        EXPECT_EQ(1, world.lv.use_count());

        // Construct proto
        ProtoConstructor make_proto(*this->volumes(), false);
        return make_proto(*world.lv);
    }

    auto get_proto_names(ProtoMap const& protos) const
    {
        StringSimplifier simplify_str;

        std::vector<std::string> result;
        for (auto uid : range(UniverseId{protos.size()}))
        {
            result.push_back(
                simplify_str(std::string(protos.at(uid)->label())));
        }
        return result;
    }

    Unit build_unit(ProtoMap const& protos, UniverseId id) const
    {
        CELER_EXPECT(id < protos.size());
        auto const* proto = dynamic_cast<UnitProto const*>(protos.at(id));
        CELER_ASSERT(proto);
        return proto->build(tol_,
                            id == UniverseId{0} ? BBox{}
                                                : BBox{{-1000, -1000, -1000},
                                                       {1000, 1000, 1000}});
    }

    Tolerance<> tol_ = Tol::from_relative(1e-5);
};

//---------------------------------------------------------------------------//
TEST_F(ProtoConstructorTest, one_box)
{
    auto global_proto = this->load("one-box");
    ProtoMap protos{*global_proto};
    ASSERT_EQ(1, protos.size());
    {
        SCOPED_TRACE("global");
        auto u = this->build_unit(protos, UniverseId{0});

        static char const* const expected_surface_strings[] = {
            "Plane: x=-500",
            "Plane: x=500",
            "Plane: y=-500",
            "Plane: y=500",
            "Plane: z=-500",
            "Plane: z=500",
        };
        static char const* const expected_volume_strings[] = {
            "!all(+0, -1, +2, -3, +4, -5)",
            "all(+0, -1, +2, -3, +4, -5)",
        };
        static char const* const expected_md_strings[] = {
            "",
            "",
            "world_box@mx",
            "world_box@px",
            "",
            "world_box@my",
            "world_box@py",
            "",
            "world_box@mz",
            "world_box@pz",
            "",
            "world_box",
            "[EXTERIOR]",
        };

        EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
        EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
        EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
        EXPECT_EQ(GeoMatId{}, u.background);
    }
}

//---------------------------------------------------------------------------//
TEST_F(ProtoConstructorTest, two_boxes)
{
    auto global_proto = this->load("two-boxes");
    ProtoMap protos{*global_proto};
    ASSERT_EQ(1, protos.size());
    {
        SCOPED_TRACE("global");
        auto u = this->build_unit(protos, UniverseId{0});

        static char const* const expected_surface_strings[] = {
            "Plane: x=-500",
            "Plane: x=500",
            "Plane: y=-500",
            "Plane: y=500",
            "Plane: z=-500",
            "Plane: z=500",
            "Plane: x=-5",
            "Plane: x=5",
            "Plane: y=-5",
            "Plane: y=5",
            "Plane: z=-5",
            "Plane: z=5",
        };
        static char const* const expected_volume_strings[] = {
            "!all(+0, -1, +2, -3, +4, -5)",
            "all(+6, -7, +8, -9, +10, -11)",
            "all(+0, -1, +2, -3, +4, -5, !all(+6, -7, +8, -9, +10, -11))",
        };
        EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
        EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
    }
}

//---------------------------------------------------------------------------//
TEST_F(ProtoConstructorTest, intersection_boxes)
{
    auto global_proto = this->load("intersection-boxes");
    ProtoMap protos{*global_proto};
    ASSERT_EQ(1, protos.size());
    {
        SCOPED_TRACE("global");
        auto u = this->build_unit(protos, UniverseId{0});

        static char const* const expected_surface_strings[] = {
            "Plane: x=-50",
            "Plane: x=50",
            "Plane: y=-50",
            "Plane: y=50",
            "Plane: z=-50",
            "Plane: z=50",
            "Plane: x=-1",
            "Plane: x=1",
            "Plane: y=-1.5",
            "Plane: y=1.5",
            "Plane: z=-2",
            "Plane: z=2",
            "Plane: n={0.86603,0,-0.5}, d=-2.6340",
            "Plane: n={0.86603,0,-0.5}, d=0.36602",
            "Plane: y=0",
            "Plane: y=4",
            "Plane: n={0.5,0,0.86603}, d=1.4641",
            "Plane: n={0.5,0,0.86603}, d=6.4641",
        };
        static char const* const expected_volume_strings[] = {
            "!all(+0, -1, +2, -3, +4, -5)",
            "all(+6, -7, +8, -9, +10, -11, +12, -13, +14, -15, +16, -17)",
            R"(all(+0, -1, +2, -3, +4, -5, !all(+6, -7, +8, -9, +10, -11, +12, -13, +14, -15, +16, -17)))"};
        static char const* const expected_md_strings[] = {
            "",
            "",
            "world_box@mx",
            "world_box@px",
            "",
            "world_box@my",
            "world_box@py",
            "",
            "world_box@mz",
            "world_box@pz",
            "",
            "world_box",
            "[EXTERIOR]",
            "first@mx",
            "first@px",
            "",
            "first@my",
            "first@py",
            "",
            "first@mz",
            "first@pz",
            "",
            "first",
            "second@mx",
            "second@px",
            "",
            "second@my",
            "second@py",
            "",
            "second@mz",
            "second@pz",
            "",
            "second",
            "isect",
            "",
            "world",
        };
        static char const* const expected_bound_strings[] = {
            "11: {{{-50,-50,-50}, {50,50,50}}, {{-50,-50,-50}, {50,50,50}}}",
            "~12: {{{-50,-50,-50}, {50,50,50}}, {{-50,-50,-50}, {50,50,50}}}",
            "22: {{{-1,-1.5,-2}, {1,1.5,2}}, {{-1,-1.5,-2}, {1,1.5,2}}}",
            "32: {null, {{-1.55,0,1.08}, {3.55,4,6.92}}}",
            "33: {null, {{-1,0,1.08}, {1,1.5,2}}}",
            "~34: {null, {{-1,0,1.08}, {1,1.5,2}}}",
            "35: {{{-1,0,1.08}, {1,1.5,2}}, {{-50,-50,-50}, {50,50,50}}}",
        };

        EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
        EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
        EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
        EXPECT_VEC_EQ(expected_bound_strings, bound_strings(u));
    }
}

//---------------------------------------------------------------------------//
TEST_F(ProtoConstructorTest, lar_sphere)
{
    auto global_proto = this->load("lar-sphere");
    ProtoMap protos{*global_proto};

    ASSERT_EQ(1, protos.size());
    {
        SCOPED_TRACE("global");
        auto u = this->build_unit(protos, UniverseId{0});

        static char const* const expected_surface_strings[] = {
            "Sphere: r=1000",
            "Sphere: r=110",
            "Sphere: r=100",
            "Plane: y=0",
        };
        static char const* const expected_volume_strings[] = {
            "+0",
            "all(-1, +2, !any(all(-1, +2, +3), all(-1, +2, -3)))",
            "all(-1, +2, +3)",
            "all(-1, +2, -3)",
            "-2",
            "all(-0, !any(-2, all(-1, +2)))",
        };
        static char const* const expected_md_strings[] = {
            "",
            "",
            "[EXTERIOR],world_sphere@s",
            "world_sphere",
            "det_shell@int.s,det_shell_bot@int.s,det_shell_top@int.s",
            "det_shell@int,det_shell_bot@int,det_shell_top@int",
            R"(det_shell@exc.s,det_shell_bot@exc.s,det_shell_top@exc.s,lar_sphere@s)",
            "det_shell@exc,det_shell_bot@exc,det_shell_top@exc,lar_sphere",
            "det_shell",
            R"(det_shell_bot@awm,det_shell_bot@awp,det_shell_top@awm,det_shell_top@awp,det_shell_top@azi)",
            "det_shell_top",
            "det_shell_bot@azi",
            "det_shell_bot",
            "detshell.children",
            "",
            "detshell",
            "world.children",
            "",
            "world",
        };
        static int const expected_volume_nodes[] = {2, 15, 10, 12, 7, 18};
        static char const expected_tree_string[]
            = R"json(["t",["~",0],["S",0],["~",2],["S",1],["~",4],["S",2],["~",6],["&",[5,6]],["S",3],["&",[5,6,9]],["~",9],["&",[5,6,11]],["|",[10,12]],["~",13],["&",[5,6,14]],["|",[7,8]],["~",16],["&",[3,17]]])json";

        EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
        EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
        EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
        EXPECT_VEC_EQ(expected_volume_nodes, volume_nodes(u));
        EXPECT_JSON_EQ(expected_tree_string, tree_string(u));
    }
}

//---------------------------------------------------------------------------//
TEST_F(ProtoConstructorTest, simple_cms)
{
    // NOTE: GDML stores widths for box and cylinder Z; Geant4 uses halfwidths
    auto global_proto = this->load("simple-cms");
    ProtoMap protos{*global_proto};

    static std::string const expected_proto_names[] = {"world"};
    EXPECT_VEC_EQ(expected_proto_names, get_proto_names(protos));

    ASSERT_EQ(1, protos.size());
    {
        SCOPED_TRACE("global");
        auto u = this->build_unit(protos, UniverseId{0});

        static char const* const expected_surface_strings[] = {
            "Plane: x=-1000",
            "Plane: x=1000",
            "Plane: y=-1000",
            "Plane: y=1000",
            "Plane: z=-2000",
            "Plane: z=2000",
            "Plane: z=-700",
            "Plane: z=700",
            "Cyl z: r=30",
            "Cyl z: r=125",
            "Cyl z: r=175",
            "Cyl z: r=275",
            "Cyl z: r=375",
            "Cyl z: r=700",
        };
        static char const* const expected_volume_strings[] = {
            "!all(+0, -1, +2, -3, +4, -5)",
            "all(+6, -7, -8)",
            "all(+6, -7, -9, !all(+6, -7, -8))",
            "all(+6, -7, -10, !all(+6, -7, -9))",
            "all(+6, -7, -11, !all(+6, -7, -10))",
            "all(+6, -7, -12, !all(+6, -7, -11))",
            "all(+6, -7, -13, !all(+6, -7, -12))",
        };
        static char const* const expected_fill_strings[]
            = {"<UNASSIGNED>", "m0", "m1", "m2", "m3", "m4", "m5"};
        static int const expected_volume_nodes[] = {12, 18, 23, 28, 33, 38, 43};

        EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
        EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
        EXPECT_VEC_EQ(expected_fill_strings, fill_strings(u));
        EXPECT_VEC_EQ(expected_volume_nodes, volume_nodes(u));
        EXPECT_EQ(GeoMatId{0}, u.background);
    }
}

//---------------------------------------------------------------------------//
TEST_F(ProtoConstructorTest, testem3)
{
    auto global_proto = this->load("testem3");
    ProtoMap protos{*global_proto};

    static std::string const expected_proto_names[] = {"world", "layer"};
    EXPECT_VEC_EQ(expected_proto_names, get_proto_names(protos));

    ASSERT_EQ(2, protos.size());
    {
        SCOPED_TRACE("global");
        auto u = this->build_unit(protos, UniverseId{0});

        // NOTE: 51 layer X surfaces, 4 surrounding, 6 world, plus whatever
        // "unused" surfaces from deduplication
        auto surfaces = surface_strings(u);
        EXPECT_LE(51 + 4 + 6, surfaces.size()) << repr(surfaces);

        auto transforms = transform_strings(u);
        EXPECT_EQ(58, transforms.size()) << repr(transforms);
        EXPECT_EQ("28: t=3 -> {{-18,0,0}}", transforms[4]);

        auto bounds = bound_strings(u);
        ASSERT_EQ(transforms.size(), bounds.size());
        EXPECT_EQ(
            "28: {{{-18.4,-20,-20}, {-17.6,20,20}}, {{-18.4,-20,-20}, "
            "{-17.6,20,20}}}",
            bounds[4]);

        auto vols = volume_strings(u);
        ASSERT_EQ(53, vols.size());  // slabs, zero-size 'calo', world, ext
        if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
        {
            // Deduplication changes for single precision
            EXPECT_EQ(
                R"(all(+0, -1, +2, -3, +4, -5, !all(+6, +8, -9, +10, -11, -84)))",
                vols.back());
        }
        EXPECT_EQ(GeoMatId{}, u.background);
    }
    {
        SCOPED_TRACE("daughter");
        auto u = this->build_unit(protos, UniverseId{1});

        static char const* const expected_surface_strings[] = {
            "Plane: x=-0.17",
        };
        static char const* const expected_volume_strings[]
            = {"F", "-6", "+6", "F"};
        static char const* const expected_md_strings[] = {
            "",
            "",
            "Absorber1@mx,Layer@mx",
            "Absorber2@px,Layer@px",
            "",
            "Absorber1@my,Absorber2@my,Layer@my",
            "Absorber1@py,Absorber2@py,Layer@py",
            "",
            "Absorber1@mz,Absorber2@mz,Layer@mz",
            "Absorber1@pz,Absorber2@pz,Layer@pz",
            "",
            "Layer",
            "[EXTERIOR]",
            "Absorber1@px,Absorber2@mx",
            "",
            "Absorber1",
            "Absorber2",
            "layer.children",
            "",
            "layer",
        };

        EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
        EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
        EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
    }
}

//---------------------------------------------------------------------------//
// Deduplication slightly changes plane position and CSG node IDs
TEST_F(ProtoConstructorTest, TEST_IF_CELERITAS_DOUBLE(tilecal_plug))
{
    auto global_proto = this->load("tilecal-plug");
    ProtoMap protos{*global_proto};

    static std::string const expected_proto_names[] = {
        "Tile_ITCModule",
    };
    EXPECT_VEC_EQ(expected_proto_names, get_proto_names(protos));

    ASSERT_EQ(1, protos.size());
    {
        auto u = this->build_unit(protos, UniverseId{0});

        static char const* const expected_surface_strings[] = {
            "Plane: z=-62.057",
            "Plane: z=62.057",
            "Plane: x=15.45",
            "Plane: n={0,0.99879,-0.049068}, d=17.711",
            "Plane: x=-15.45",
            "Plane: n={0,0.99879,0.049068}, d=-17.711",
            "Plane: z=-16.942",
            "Plane: z=-17.058",
            "Plane: x=5.965",
            "Plane: z=25.058",
            "Plane: n={0,0.99879,-0.049068}, d=17.636",
            "Plane: n={0,0.99879,0.049068}, d=-17.636",
        };
        static char const* const expected_fill_strings[]
            = {"<UNASSIGNED>", "m1", "m0", "m1"};
        static int const expected_volume_nodes[] = {12, 28, 26, 30};
        static char const expected_tree_string[]
            = R"json(["t",["~",0],["S",0],["S",1],["~",3],["S",2],["~",5],["S",3],["~",7],["S",4],["S",5],["&",[2,4,6,8,9,10]],["~",11],["S",6],["&",[4,6,8,9,10,13]],["S",9],["~",13],["S",10],["~",17],["&",[8,9,10,15,16,18]],["|",[14,19]],["S",13],["~",21],["S",14],["~",23],["S",15],["&",[6,9,13,22,24,25]],["~",26],["&",[20,27]],["~",20],["&",[2,4,6,8,9,10,29]]])json";

        EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
        EXPECT_VEC_EQ(expected_fill_strings, fill_strings(u));
        EXPECT_VEC_EQ(expected_volume_nodes, volume_nodes(u));
        EXPECT_JSON_EQ(expected_tree_string, tree_string(u));
        EXPECT_EQ(GeoMatId{}, u.background);
    }
}

//---------------------------------------------------------------------------//
TEST_F(ProtoConstructorTest, znenv)
{
    auto global_proto = this->load("znenv");
    ProtoMap protos{*global_proto};

    static std::string const expected_proto_names[] = {
        "World",
        "ZNTX",
        "ZN1",
        "ZNSL",
        "ZNST",
        "ZNG1",
        "ZNG2",
        "ZNG3",
        "ZNG4",
    };
    EXPECT_VEC_EQ(expected_proto_names, get_proto_names(protos));

    ASSERT_EQ(9, protos.size());
    {
        SCOPED_TRACE("World");
        auto u = this->build_unit(protos, UniverseId{0});

        // clang-format off
        static char const* const expected_surface_strings[] = {
            "Plane: x=-50",   "Plane: x=50",    "Plane: y=-50",
            "Plane: y=50",    "Plane: z=-100",  "Plane: z=100",
            "Plane: x=-3.52", "Plane: x=0",     "Plane: y=-3.52",
            "Plane: y=3.52",  "Plane: z=-50",   "Plane: z=50",
            "Plane: x=3.52",  "Plane: x=-3.62", "Plane: x=3.62",
            "Plane: y=-3.62", "Plane: y=3.62",  "Plane: z=-50.1",
            "Plane: z=50.1",
        };
        static char const* const expected_fill_strings[] = {
            "<UNASSIGNED>",
            "{u=0, t=1}",
            "{u=1, t=2}",
            "m3",
            "m2",
            "m3",
        };
        static int const expected_volume_nodes[] = {12, 22, 25, 38, 41, 43};
        static char const expected_tree_string[]
            = R"json(["t",["~",0],["S",0],["S",1],["~",3],["S",2],["S",3],["~",6],["S",4],["S",5],["~",9],["&",[2,4,5,7,8,10]],["~",11],["S",6],["S",7],["~",14],["S",8],["S",9],["~",17],["S",10],["S",11],["~",20],["&",[13,15,16,18,19,21]],["S",12],["~",23],["&",[14,16,18,19,21,24]],["S",13],["S",14],["~",27],["S",15],["S",16],["~",30],["S",17],["S",18],["~",33],["&",[26,28,29,31,32,34]],["&",[13,16,18,19,21,24]],["~",36],["&",[26,28,29,31,32,34,37]],["|",[22,25]],["~",39],["&",[13,16,18,19,21,24,40]],["~",35],["&",[2,4,5,7,8,10,42]]])json";
        // clang-format on

        EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
        EXPECT_VEC_EQ(expected_fill_strings, fill_strings(u));
        EXPECT_VEC_EQ(expected_volume_nodes, volume_nodes(u));
        EXPECT_JSON_EQ(expected_tree_string, tree_string(u));
        EXPECT_EQ(GeoMatId{}, u.background);
    }
    {
        SCOPED_TRACE("ZNTX");
        auto u = this->build_unit(protos, UniverseId{1});

        static char const* const expected_surface_strings[] = {
            "Plane: y=0",
        };
        static char const* const expected_volume_strings[]
            = {"F", "-6", "+6", "F"};

        EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
        EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
        EXPECT_EQ(GeoMatId{}, u.background);
    }
    {
        SCOPED_TRACE("ZNST");
        auto u = this->build_unit(protos, UniverseId{4});

        static char const* const expected_surface_strings[] = {
            "Plane: x=-0.11",
            "Plane: x=-0.05",
            "Plane: y=0.05",
            "Plane: y=0.11",
            "Plane: x=0.05",
            "Plane: x=0.11",
            "Plane: y=-0.11",
            "Plane: y=-0.05",
        };
        static char const* const expected_volume_strings[] = {
            "F",
            "all(+6, -7, +8, -9)",
            "all(+8, -9, +10, -11)",
            "all(+6, -7, +12, -13)",
            "all(+10, -11, +12, -13)",
        };

        EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
        EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
        EXPECT_EQ(GeoMatId{2}, u.background);
    }
    {
        SCOPED_TRACE("ZNG1");
        auto u = this->build_unit(protos, UniverseId{5});
        static char const* const expected_surface_strings[]
            = {"Cyl z: r=0.01825"};
        static char const* const expected_volume_strings[] = {"F", "-6", "+6"};

        EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
        EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
        EXPECT_EQ(GeoMatId{}, u.background);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace g4org
}  // namespace celeritas
