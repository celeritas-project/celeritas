//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/g4org/ProtoConstructor.test.cc
//---------------------------------------------------------------------------//
#include "orange/g4org/ProtoConstructor.hh"

#include "corecel/Config.hh"

#include "corecel/OpaqueIdUtils.hh"
#include "corecel/StringSimplifier.hh"
#include "corecel/io/Repr.hh"
#include "geocel/GeantGeoParams.hh"
#include "geocel/VolumeToString.hh"
#include "orange/OrangeData.hh"
#include "orange/OrangeTypes.hh"
#include "orange/g4org/PhysicalVolumeConverter.hh"
#include "orange/orangeinp/CsgTestUtils.hh"
#include "orange/orangeinp/UnitProto.hh"
#include "orange/orangeinp/detail/CsgUnit.hh"
#include "orange/orangeinp/detail/ProtoMap.hh"

#include "GeantLoadTestBase.hh"
#include "TestMacros.hh"
#include "celeritas_test.hh"

using namespace celeritas::orangeinp::test;
using celeritas::orangeinp::UnitProto;
using celeritas::orangeinp::detail::ProtoMap;
using celeritas::test::id_to_int;
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
    using Options = inp::OrangeGeoFromGeant;

    void SetUp() override
    {
        opts_.unit_length = 0.1;
        opts_.tol = Tolerance<>::from_relative(1e-5);
    }

    void setup_opts(std::string const& json_str)
    {
        CELER_EXPECT(!json_str.empty());
        std::istringstream{json_str} >> opts_;

        // Check JSON
        auto&& result_str = [&opts = opts_] {
            std::ostringstream os;
            os << opts;
            return std::move(os).str();
        }();
        EXPECT_EQ(15, std::count(result_str.begin(), result_str.end(), ','))
            << "JSON items changed: actual is " << repr(result_str);
    }

    std::shared_ptr<UnitProto> load(std::string const& basename)
    {
        // Load GDML into Geant4
        this->load_test_gdml(basename);

        // Convert volumes into ORANGE representation
        auto const& geant_geo = this->geo();
        PhysicalVolumeConverter make_pv(geant_geo, opts_);
        PhysicalVolume world = make_pv(*geant_geo.world());

        EXPECT_TRUE(std::holds_alternative<NoTransformation>(world.transform));
        EXPECT_EQ(1, world.lv.use_count());

        // Construct proto
        ProtoConstructor make_proto(*this->volumes(), opts_);
        return make_proto(*world.lv);
    }

    auto get_proto_names(ProtoMap const& protos) const
    {
        StringSimplifier simplify_str;

        std::vector<std::string> result;
        for (auto uid : range(UnivId{protos.size()}))
        {
            result.push_back(
                simplify_str(std::string(protos.at(uid)->label())));
        }
        return result;
    }

    auto get_all_local_parents(ProtoMap const& protos) const
    {
        VolumeToString to_string{*this->volumes()};
        std::vector<std::vector<std::string>> result;
        for (auto id : range(UnivId{protos.size()}))
        {
            std::vector<std::string> local_parents;
            if (auto const* unit = dynamic_cast<UnitProto const*>(protos.at(id)))
            {
                auto const& local_mats = unit->input().materials;
                for (auto const& mat : unit->input().materials)
                {
                    if (mat.local_parent)
                    {
                        // Optional opaque ID is given
                        std::string s = std::visit(to_string, mat.label);
                        s += ',';
                        if (auto lp_id = *mat.local_parent)
                        {
                            CELER_ASSERT(lp_id < local_mats.size());
                            auto const& parent_mat = local_mats[lp_id.get()];
                            s += std::visit(to_string, parent_mat.label);
                        }
                        else
                        {
                            auto bg = unit->input().background.label;
                            s += "bg@";
                            s += unit->label();
                            s += '=';
                            s += std::visit(to_string, bg);
                        }

                        local_parents.emplace_back(std::move(s));
                    }
                }
            }
            result.push_back(std::move(local_parents));
        }
        return result;
    }

    Unit build_unit(ProtoMap const& protos, UnivId id) const
    {
        CELER_EXPECT(id < protos.size());
        auto const* proto = dynamic_cast<UnitProto const*>(protos.at(id));
        CELER_ASSERT(proto);
        UnitProto::BuildOptions opts;
        opts.assume_inside = opts_.implicit_parent_boundary
                             && (id != orange_global_univ);
        opts.tol = opts_.tol;
        opts.logic = LogicNotation::infix;
        return proto->build(opts);
    }

  private:
    Options opts_;
};

//---------------------------------------------------------------------------//
using AtlasLarEndcapTest = ProtoConstructorTest;

TEST_F(AtlasLarEndcapTest, default)
{
    auto global_proto = this->load("atlas-lar-endcap");
    ProtoMap protos{*global_proto};
    ASSERT_EQ(1, protos.size());
    {
        SCOPED_TRACE("global");
        auto u = this->build_unit(protos, UnivId{0});

        EXPECT_JSON_EQ(R"json({"czc": 2, "gq": 25, "p": 50, "pz": 2})json",
                       count_surface_types(u));
    }
}

//---------------------------------------------------------------------------//
using DuneCryostatTest = ProtoConstructorTest;

TEST_F(DuneCryostatTest, default)
{
    auto global_proto = this->load("dune-cryostat");
    ProtoMap protos{*global_proto};
    EXPECT_EQ(2, protos.size());
    {
        SCOPED_TRACE("enclosure");
        auto u = this->build_unit(protos, UnivId{0});

        static char const* const expected_surface_strings[] = {
            "Plane: x=-609.66",
            "Plane: x=609.66",
            "Plane: y=-908.85",
            "Plane: y=908.85",
            "Plane: z=-1153.2",
            "Plane: z=1153.2",
            "Plane: x=-379.66",
            "Plane: x=379.66",
            "Plane: y=-778.85",
            "Plane: y=578.85",
            "Plane: z=-873.24",
            "Plane: z=873.24",
        };
        static char const* const expected_volume_strings[] = {
            "any(-0, +1, -2, +3, -4, +5)",
            "all(+6, -7, +8, -9, +10, -11)",
            "all(+0, -1, +2, -3, +4, -5, any(-6, +7, -8, +9, -10, +11))",
        };
        static char const* const expected_md_strings[] = {
            "",
            "",
            "DetEnclosure@mx",
            "",
            "DetEnclosure@px",
            "",
            "DetEnclosure@my",
            "",
            "DetEnclosure@py",
            "",
            "DetEnclosure@mz",
            "",
            "DetEnclosure@pz",
            "",
            "[EXTERIOR]",
            "Cryostat@mx",
            "",
            "Cryostat@px",
            "",
            "Cryostat@my",
            "",
            "Cryostat@py",
            "",
            "Cryostat@mz",
            "",
            "Cryostat@pz",
            "",
            "",
            "Cryostat",
            "volDetEnclosure",
        };
        EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
        EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
        EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
    }
    {
        SCOPED_TRACE("cryostat");
        // NOTE: volume 2 (CSG node 52 = ArapucaWalls) is a series of
        // subtractions that results in an inf/null bzone: this needs to be
        // fixed
        auto u = this->build_unit(protos, UnivId{1});

        static char const* const expected_surface_strings[] = {
            "Plane: x=-378.39", "Plane: x=378.39",  "Plane: y=627.58",
            "Plane: y=677.58",  "Plane: z=-871.97", "Plane: z=871.97",
            "Plane: x=-1.15",   "Plane: x=1.15",    "Plane: y=-618.22",
            "Plane: y=-606.42", "Plane: z=-559.58", "Plane: z=-350.38",
            "Plane: x=-1.2",    "Plane: x=1.2",     "Plane: y=-616.97",
            "Plane: y=-607.67", "Plane: z=-558.58", "Plane: z=-511.77",
            "Plane: z=-509.77", "Plane: z=-462.98", "Plane: z=-446.98",
            "Plane: z=-400.18", "Plane: z=-398.18", "Plane: z=-351.38",
            "Plane: x=1.05",
        };
        static char const* const expected_volume_strings[] = {
            "F",
            "all(+6, -7, +8, -9, +10, -11)",
            R"(all(+12, -13, +14, -15, +16, -17, any(-18, +19, -20, +21, -22, +23), any(-18, +19, -20, +21, -24, +25), any(-18, +19, -20, +21, -26, +27), any(-18, +19, -20, +21, -28, +29)))",
            "all(+12, +20, -21, +22, -23, -31)",
            "all(+12, +20, -21, +24, -25, -31)",
            "all(+12, +20, -21, +26, -27, -31)",
            "all(+12, +20, -21, +28, -29, -31)",
        };
        static char const* const expected_md_strings[] = {
            "Cryostat,Cryostat@mx,Cryostat@my,Cryostat@mz",
            "Cryostat@px,Cryostat@py,Cryostat@pz,[EXTERIOR]",
            "GaseousArgon@mx",
            "GaseousArgon@px",
            "",
            "GaseousArgon@my",
            "GaseousArgon@py",
            "",
            "GaseousArgon@mz",
            "GaseousArgon@pz",
            "",
            "GaseousArgon",
            "ArapucaAcceptanceWindow@mx,ArapucaOut@mx",
            "ArapucaOut@px",
            "",
            "ArapucaOut@my",
            "ArapucaOut@py",
            "",
            "ArapucaOut@mz",
            "ArapucaOut@pz",
            "",
            "ArapucaOut",
            "ArapucaIn@mx",
            "",
            "ArapucaIn@px",
            "ArapucaAcceptanceWindow@my,ArapucaIn@my",
            "",
            "ArapucaAcceptanceWindow@py,ArapucaIn@py",
            "",
            "ArapucaAcceptanceWindow@mz,ArapucaIn@mz",
            "",
            "ArapucaAcceptanceWindow@pz,ArapucaIn@pz",
            "",
            "",
            "ArapucaWalls0",
            "ArapucaAcceptanceWindow@mz,ArapucaIn@mz",
            "",
            "ArapucaAcceptanceWindow@pz,ArapucaIn@pz",
            "",
            "",
            "ArapucaWalls1",
            "ArapucaAcceptanceWindow@mz,ArapucaIn@mz",
            "",
            "ArapucaAcceptanceWindow@pz,ArapucaIn@pz",
            "",
            "",
            "ArapucaWalls2",
            "ArapucaAcceptanceWindow@mz,ArapucaIn@mz",
            "",
            "ArapucaAcceptanceWindow@pz,ArapucaIn@pz",
            "",
            "",
            "ArapucaWalls",
            "ArapucaAcceptanceWindow@px",
            "",
            "ArapucaAcceptanceWindow",
            "ArapucaAcceptanceWindow",
            "ArapucaAcceptanceWindow",
            "ArapucaAcceptanceWindow",
        };
        static char const* const expected_bound_strings[] = {
            R"(0: {{{-380,-679,-873}, {380,679,873}}, {{-380,-679,-873}, {380,679,873}}})",
            R"(~1: {{{-380,-679,-873}, {380,679,873}}, {{-380,-679,-873}, {380,679,873}}})",
            R"(11: {{{-378,628,-872}, {378,678,872}}, {{-378,628,-872}, {378,678,872}}})",
            R"(21: {{{-1.15,-618,-560}, {1.15,-606,-350}}, {{-1.15,-618,-560}, {1.15,-606,-350}}})",
            R"(~33: {{{-1.2,-617,-559}, {1.2,-608,-512}}, {{-1.2,-617,-559}, {1.2,-608,-512}}})",
            "34: {null, inf}",
            R"(~39: {{{-1.2,-617,-510}, {1.2,-608,-463}}, {{-1.2,-617,-510}, {1.2,-608,-463}}})",
            "40: {null, inf}",
            R"(~45: {{{-1.2,-617,-447}, {1.2,-608,-400}}, {{-1.2,-617,-447}, {1.2,-608,-400}}})",
            "46: {null, inf}",
            R"(~51: {{{-1.2,-617,-398}, {1.2,-608,-351}}, {{-1.2,-617,-398}, {1.2,-608,-351}}})",
            "52: {null, inf}",
            R"(55: {{{-1.15,-617,-559}, {1.05,-608,-512}}, {{-1.15,-617,-559}, {1.05,-608,-512}}})",
            R"(56: {{{-1.15,-617,-510}, {1.05,-608,-463}}, {{-1.15,-617,-510}, {1.05,-608,-463}}})",
            R"(57: {{{-1.15,-617,-447}, {1.05,-608,-400}}, {{-1.15,-617,-447}, {1.05,-608,-400}}})",
            R"(58: {{{-1.15,-617,-398}, {1.05,-608,-351}}, {{-1.15,-617,-398}, {1.05,-608,-351}}})",
        };
        static char const* const expected_trans_strings[] = {
            "0: t=0 -> {}",
            "1: t=0",
            "11: t=1 -> {{0,653,0}}",
            "21: t=2 -> {{0,-612,-455}}",
            "33: t=2",
            "34: t=2",
            "39: t=2",
            "40: t=2",
            "45: t=2",
            "46: t=2",
            "51: t=2",
            "52: t=2",
            "55: t=7 -> {{-0.05,-612,-535}}",
            "56: t=8 -> {{-0.05,-612,-486}}",
            "57: t=9 -> {{-0.05,-612,-424}}",
            "58: t=10 -> {{-0.05,-612,-375}}",
        };
        static char const* const expected_fill_strings[]
            = {"<UNASSIGNED>", "m0", "m1", "m2", "m2", "m2", "m2"};
        static int const expected_volume_nodes[] = {1, 11, 52, 55, 56, 57, 58};
        static char const expected_tree_string[]
            = R"json(["t",["~",0],["S",6],["S",7],["~",3],["S",8],["S",9],["~",6],["S",10],["S",11],["~",9],["&",[2,4,5,7,8,10]],["S",12],["S",13],["~",13],["S",14],["S",15],["~",16],["S",16],["S",17],["~",19],["&",[12,14,15,17,18,20]],["S",18],["~",22],["S",19],["S",20],["~",25],["S",21],["~",27],["S",22],["~",29],["S",23],["~",31],["|",[23,24,26,27,30,31]],["&",[12,14,15,17,18,20,33]],["S",24],["~",35],["S",25],["~",37],["|",[23,24,26,27,36,37]],["&",[12,14,15,17,18,20,33,39]],["S",26],["~",41],["S",27],["~",43],["|",[23,24,26,27,42,43]],["&",[12,14,15,17,18,20,33,39,45]],["S",28],["~",47],["S",29],["~",49],["|",[23,24,26,27,48,49]],["&",[12,14,15,17,18,20,33,39,45,51]],["S",31],["~",53],["&",[12,25,28,29,32,54]],["&",[12,25,28,35,38,54]],["&",[12,25,28,41,44,54]],["&",[12,25,28,47,50,54]]])json";

        EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
        EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
        EXPECT_VEC_EQ(expected_bound_strings, bound_strings(u));
        EXPECT_VEC_EQ(expected_trans_strings, transform_strings(u));
        EXPECT_VEC_EQ(expected_fill_strings, fill_strings(u));
        EXPECT_VEC_EQ(expected_volume_nodes, volume_nodes(u));
        if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
        {
            // Deduplication changes for single precision
            EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
            EXPECT_JSON_EQ(expected_tree_string, tree_string(u));
        }
        EXPECT_EQ(GeoMatId{3}, u.background);
    }
}

//---------------------------------------------------------------------------//
using IntersectionBoxesTest = ProtoConstructorTest;

TEST_F(IntersectionBoxesTest, default)
{
    auto global_proto = this->load("intersection-boxes");
    ProtoMap protos{*global_proto};
    ASSERT_EQ(1, protos.size());
    {
        SCOPED_TRACE("global");
        auto u = this->build_unit(protos, UnivId{0});

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
            "any(-0, +1, -2, +3, -4, +5)",
            "all(+6, -7, +8, -9, +10, -11, +12, -13, +14, -15, +16, -17)",
            R"(all(+0, -1, +2, -3, +4, -5, any(-6, +7, -8, +9, -10, +11, -12, +13, -14, +15, -16, +17)))"};
        static char const* const expected_md_strings[] = {
            "",
            "",
            "world_box@mx",
            "",
            "world_box@px",
            "",
            "world_box@my",
            "",
            "world_box@py",
            "",
            "world_box@mz",
            "",
            "world_box@pz",
            "",
            "[EXTERIOR]",
            "first@mx",
            "",
            "first@px",
            "",
            "first@my",
            "",
            "first@py",
            "",
            "first@mz",
            "",
            "first@pz",
            "",
            "first",
            "second@mx",
            "",
            "second@px",
            "",
            "second@my",
            "",
            "second@py",
            "",
            "second@mz",
            "",
            "second@pz",
            "",
            "second",
            "",
            "isect",
            "world",
        };
        static char const* const expected_bound_strings[] = {
            "~14: {{{-50,-50,-50}, {50,50,50}}, {{-50,-50,-50}, {50,50,50}}}",
            "27: {{{-1,-1.5,-2}, {1,1.5,2}}, {{-1,-1.5,-2}, {1,1.5,2}}}",
            "40: {null, {{-1.55,0,1.08}, {3.55,4,6.92}}}",
            "~41: {null, {{-1,0,1.08}, {1,1.5,2}}}",
            "42: {null, {{-1,0,1.08}, {1,1.5,2}}}",
            "43: {{{-1,0,1.08}, {1,1.5,2}}, {{-50,-50,-50}, {50,50,50}}}",
        };

        EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
        EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
        EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
        EXPECT_VEC_EQ(expected_bound_strings, bound_strings(u));
    }
}

//---------------------------------------------------------------------------//
using LarSphereTest = ProtoConstructorTest;

TEST_F(LarSphereTest, default)
{
    auto global_proto = this->load("lar-sphere");
    ProtoMap protos{*global_proto};

    ASSERT_EQ(1, protos.size());
    {
        SCOPED_TRACE("global");
        auto u = this->build_unit(protos, UnivId{0});

        static char const* const expected_surface_strings[] = {
            "Sphere: r=1000",
            "Sphere: r=110",
            "Sphere: r=100",
            "Plane: y=0",
        };
        static char const* const expected_volume_strings[] = {
            "+0",
            "all(-1, +2, any(+1, -2, -3), any(+1, -2, +3))",
            "all(-1, +2, +3)",
            "all(-1, +2, -3)",
            "-2",
            "all(-0, +2, any(+1, -2))",
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
            "",
            R"(det_shell_bot@awm,det_shell_bot@awp,det_shell_top@awm,det_shell_top@awp,det_shell_top@azi)",
            "det_shell_bot@azi",
            "",
            "det_shell_top",
            "",
            "det_shell_bot",
            "",
            "detshell",
            "",
            "world",
        };
        static int const expected_volume_nodes[] = {2, 16, 12, 14, 7, 18};
        static char const expected_tree_string[]
            = R"json(["t",["~",0],["S",0],["~",2],["S",1],["~",4],["S",2],["~",6],["|",[4,7]],["S",3],["~",9],["|",[4,7,10]],["&",[5,6,9]],["|",[4,7,9]],["&",[5,6,10]],["&",[11,13]],["&",[5,6,11,13]],["&",[6,8]],["&",[3,6,8]]])json";

        EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
        EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
        EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
        EXPECT_VEC_EQ(expected_volume_nodes, volume_nodes(u));
        EXPECT_JSON_EQ(expected_tree_string, tree_string(u));
    }
}

//----------------------------------------------------------------------------//
using MultilevelTest = ProtoConstructorTest;

TEST_F(MultilevelTest, full_inline)
{
    this->setup_opts(R"json({
"_format": "g4org-options",
"explicit_interior_threshold": 1000,
"inline_childless": true,
"inline_singletons": "all",
"inline_unions": true,
"verbose_structure": false
})json");

    auto global_proto = this->load("multi-level");
    ProtoMap protos{*global_proto};
    auto parents = this->get_all_local_parents(protos);

    std::vector<std::string> const expected_parents[] = {
        {"topsph1,bg@world=",
         "topbox4,bg@world=",
         "boxsph1@1,topbox4",
         "boxsph2@1,topbox4",
         "boxtri@1,topbox4"},
        {"boxsph1@0,bg@box=", "boxsph2@0,bg@box=", "boxtri@0,bg@box="},
    };
    EXPECT_VEC_EQ(expected_parents, parents);
}

TEST_F(MultilevelTest, default)
{
    auto global_proto = this->load("multi-level");
    ProtoMap protos{*global_proto};
    auto parents = this->get_all_local_parents(protos);
    std::vector<std::string> const expected_parents[] = {
        {"topsph1,bg@world=<null>"},
        {"boxsph1@0,bg@box=<null>",
         "boxsph2@0,bg@box=<null>",
         "boxtri@0,bg@box=<null>"},
        {"boxsph1@1,bg@box_refl=<null>",
         "boxsph2@1,bg@box_refl=<null>",
         "boxtri@1,bg@box_refl=<null>"},
    };
    EXPECT_VEC_EQ(expected_parents, parents);
}

TEST_F(MultilevelTest, each_volume)
{
    this->setup_opts(R"json({
"_format": "g4org-options",
"explicit_interior_threshold": 0,
"inline_childless": false,
"inline_singletons": "none",
"inline_unions": false,
"implicit_parent_boundary": false,
"logic": "infix",
"verbose_structure": false
})json");

    auto global_proto = this->load("multi-level");
    ProtoMap protos{*global_proto};
    auto parents = this->get_all_local_parents(protos);
    std::vector<std::string> const expected_parents[]
        = {{}, {}, {}, {}, {}, {}, {}};
    EXPECT_VEC_EQ(expected_parents, parents);
}

//---------------------------------------------------------------------------//
using OneBoxTest = ProtoConstructorTest;

TEST_F(OneBoxTest, default)
{
    auto global_proto = this->load("one-box");
    ProtoMap protos{*global_proto};
    ASSERT_EQ(1, protos.size());
    {
        SCOPED_TRACE("global");
        auto u = this->build_unit(protos, UnivId{0});

        static char const* const expected_surface_strings[] = {
            "Plane: x=-500",
            "Plane: x=500",
            "Plane: y=-500",
            "Plane: y=500",
            "Plane: z=-500",
            "Plane: z=500",
        };
        static char const* const expected_volume_strings[] = {
            "any(-0, +1, -2, +3, -4, +5)",
            "all(+0, -1, +2, -3, +4, -5)",
        };
        static char const* const expected_md_strings[] = {
            "",
            "",
            "world_box@mx",
            "",
            "world_box@px",
            "",
            "world_box@my",
            "",
            "world_box@py",
            "",
            "world_box@mz",
            "",
            "world_box@pz",
            "",
            "[EXTERIOR]",
            "world_box",
        };

        EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
        EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
        EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
        EXPECT_EQ(GeoMatId{}, u.background);
    }
}

//---------------------------------------------------------------------------//
using SimpleCmsTest = ProtoConstructorTest;

TEST_F(SimpleCmsTest, default)
{
    // NOTE: GDML stores widths for box and cylinder Z; Geant4 uses halfwidths
    auto global_proto = this->load("simple-cms");
    ProtoMap protos{*global_proto};

    static std::string const expected_proto_names[] = {"world"};
    EXPECT_VEC_EQ(expected_proto_names, get_proto_names(protos));

    ASSERT_EQ(1, protos.size());
    {
        SCOPED_TRACE("global");
        auto u = this->build_unit(protos, UnivId{0});

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
            "any(-0, +1, -2, +3, -4, +5)",
            "all(+6, -7, -8)",
            "all(+6, -7, any(-6, +7, +8), -9)",
            "all(+6, -7, any(-6, +7, +9), -10)",
            "all(+6, -7, any(-6, +7, +10), -11)",
            "all(+6, -7, any(-6, +7, +11), -12)",
            "all(+6, -7, any(-6, +7, +12), -13)",
        };
        static char const* const expected_fill_strings[]
            = {"<UNASSIGNED>", "m0", "m1", "m2", "m3", "m4", "m5"};
        static int const expected_volume_nodes[] = {11, 19, 23, 27, 31, 35, 39};

        EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
        EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
        EXPECT_VEC_EQ(expected_fill_strings, fill_strings(u));
        EXPECT_VEC_EQ(expected_volume_nodes, volume_nodes(u));
        EXPECT_EQ(GeoMatId{0}, u.background);
    }
}

//---------------------------------------------------------------------------//
using Testem3Test = ProtoConstructorTest;

TEST_F(Testem3Test, default)
{
    auto global_proto = this->load("testem3");
    ProtoMap protos{*global_proto};

    static std::string const expected_proto_names[] = {"world", "layer"};
    EXPECT_VEC_EQ(expected_proto_names, get_proto_names(protos));

    ASSERT_EQ(2, protos.size());
    {
        SCOPED_TRACE("global");
        auto u = this->build_unit(protos, UnivId{0});

        // NOTE: 51 layer X surfaces, 4 surrounding, 6 world, plus whatever
        // "unused" surfaces from deduplication
        auto surfaces = surface_strings(u);
        EXPECT_LE(51 + 4 + 6, surfaces.size()) << repr(surfaces);

        auto transforms = transform_strings(u);
        EXPECT_EQ(55, transforms.size()) << repr(transforms);
        EXPECT_EQ("40: t=4 -> {{-17.2,0,0}}", transforms[4]);

        auto bounds = bound_strings(u);
        ASSERT_EQ(transforms.size(), bounds.size());
        EXPECT_EQ(
            R"(40: {{{-17.6,-20,-20}, {-16.8,20,20}}, {{-17.6,-20,-20}, {-16.8,20,20}}})",
            bounds[4]);

        auto vols = volume_strings(u);
        ASSERT_EQ(53, vols.size());  // slabs, zero-size 'calo', world, ext
        if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
        {
            // Deduplication changes for single precision
            EXPECT_EQ(
                R"(all(+0, -1, +2, -3, +4, -5, any(-6, -8, +9, -10, +11, +84)))",
                vols.back());
        }
        EXPECT_EQ(GeoMatId{}, u.background);
    }
    {
        SCOPED_TRACE("daughter");
        auto u = this->build_unit(protos, UnivId{1});

        static char const* const expected_surface_strings[] = {
            "Plane: x=-0.17",
        };
        static char const* const expected_volume_strings[]
            = {"F", "-6", "+6", "F"};
        static char const* const expected_md_strings[] = {
            R"(Absorber1@mx,Absorber1@my,Absorber1@mz,Absorber2@my,Absorber2@mz,Layer,Layer@mx,Layer@my,Layer@mz,layer.children)",
            R"(Absorber1@py,Absorber1@pz,Absorber2@px,Absorber2@py,Absorber2@pz,Layer@px,Layer@py,Layer@pz,[EXTERIOR],layer)",
            "Absorber1@px,Absorber2,Absorber2@mx",
            "Absorber1",
        };

        EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
        EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
        EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
    }
}

//---------------------------------------------------------------------------//
using TilecalPlugTest = ProtoConstructorTest;

TEST_F(TilecalPlugTest, default)
{
    auto global_proto = this->load("tilecal-plug");
    ProtoMap protos{*global_proto};

    static std::string const expected_proto_names[] = {
        "Tile_ITCModule",
    };
    EXPECT_VEC_EQ(expected_proto_names, get_proto_names(protos));

    ASSERT_EQ(1, protos.size());

    if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_FLOAT)
    {
        GTEST_SKIP() << "Deduplication slightly changes surface nodes";
    }

    {
        auto u = this->build_unit(protos, UnivId{0});

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
        static int const expected_volume_nodes[] = {14, 35, 34, 36};
        static char const expected_tree_string[]
            = R"json(["t",["~",0],["S",0],["~",2],["S",1],["~",4],["S",2],["~",6],["S",3],["~",8],["S",4],["~",10],["S",5],["~",12],["|",[3,4,6,8,11,13]],["S",6],["~",15],["|",[4,6,8,11,13,16]],["&",[5,7,9,10,12,15]],["S",9],["~",19],["S",10],["~",21],["|",[8,11,13,15,20,21]],["&",[9,10,12,16,19,22]],["&",[17,23]],["|",[18,24]],["S",13],["~",27],["S",14],["~",29],["S",15],["~",31],["|",[6,11,16,27,29,32]],["&",[7,10,15,28,30,31]],["&",[26,33]],["&",[2,5,7,9,10,12,17,23]]])json";

        EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
        EXPECT_VEC_EQ(expected_fill_strings, fill_strings(u));
        EXPECT_VEC_EQ(expected_volume_nodes, volume_nodes(u));
        EXPECT_JSON_EQ(expected_tree_string, tree_string(u));
        EXPECT_EQ(GeoMatId{}, u.background);
    }
}

//---------------------------------------------------------------------------//
using TwoBoxesTest = ProtoConstructorTest;

TEST_F(TwoBoxesTest, default)
{
    auto global_proto = this->load("two-boxes");
    ProtoMap protos{*global_proto};
    ASSERT_EQ(1, protos.size());
    {
        SCOPED_TRACE("global");
        auto u = this->build_unit(protos, UnivId{0});

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
            "any(-0, +1, -2, +3, -4, +5)",
            "all(+6, -7, +8, -9, +10, -11)",
            "all(+0, -1, +2, -3, +4, -5, any(-6, +7, -8, +9, -10, +11))",
        };
        EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
        EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
    }
}

//---------------------------------------------------------------------------//
using ZnenvTest = ProtoConstructorTest;

TEST_F(ZnenvTest, default)
{
    auto global_proto = this->load("znenv");
    ProtoMap protos{*global_proto};

    static char const* const expected_proto_names[] = {
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
        auto u = this->build_unit(protos, UnivId{0});

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
        static int const expected_volume_nodes[] = {14, 28, 32, 47, 49, 50};
        static char const expected_tree_string[]
            = R"json(["t",["~",0],["S",0],["~",2],["S",1],["~",4],["S",2],["~",6],["S",3],["~",8],["S",4],["~",10],["S",5],["~",12],["|",[3,4,7,8,11,12]],["S",6],["~",15],["S",7],["~",17],["S",8],["~",19],["S",9],["~",21],["S",10],["~",23],["S",11],["~",25],["|",[16,17,20,21,24,25]],["&",[15,18,19,22,23,26]],["S",12],["~",29],["|",[18,20,21,24,25,29]],["&",[17,19,22,23,26,30]],["S",13],["~",33],["S",14],["~",35],["S",15],["~",37],["S",16],["~",39],["S",17],["~",41],["S",18],["~",43],["|",[34,35,38,39,42,43]],["|",[16,20,21,24,25,29]],["&",[33,36,37,40,41,44,46]],["&",[27,31]],["&",[15,19,22,23,26,27,30,31]],["&",[2,5,6,9,10,13,45]]])json";
        // clang-format on

        EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
        EXPECT_VEC_EQ(expected_fill_strings, fill_strings(u));
        EXPECT_VEC_EQ(expected_volume_nodes, volume_nodes(u));
        EXPECT_JSON_EQ(expected_tree_string, tree_string(u));
        EXPECT_EQ(GeoMatId{}, u.background);
    }
    {
        // Lots of eliminations happen in ZNTX
        SCOPED_TRACE("ZNTX");

        static char const* const expected_surface_strings[] = {"Plane: y=0"};
        static char const* const expected_volume_strings[]
            = {"F", "-6", "+6", "F"};
        static char const* const expected_md_strings[] = {
            R"(TGeoBBox0x0,TGeoBBox0x0@mx,TGeoBBox0x0@my,TGeoBBox0x0@mz,TGeoBBox0x0@mx,TGeoBBox0x0@my,TGeoBBox0x0@mz,ZNTX.children)",
            R"(TGeoBBox0x0@px,TGeoBBox0x0@py,TGeoBBox0x0@pz,TGeoBBox0x0@px,TGeoBBox0x0@py,TGeoBBox0x0@pz,ZNTX,[EXTERIOR])",
            "TGeoBBox0x0,TGeoBBox0x0@my,TGeoBBox0x0@py",
            "TGeoBBox0x0",
        };
        static char const* const expected_bound_strings[] = {
            R"(0: {{{-1.76,-3.52,-50}, {1.76,3.52,50}}, {{-1.76,-3.52,-50}, {1.76,3.52,50}}})",
            R"(~1: {{{-1.76,-3.52,-50}, {1.76,3.52,50}}, {{-1.76,-3.52,-50}, {1.76,3.52,50}}})",
            R"(2: {{{-1.76,0,-50}, {1.76,3.52,50}}, {{-1.76,0,-50}, {1.76,3.52,50}}})",
            R"(3: {{{-1.76,-3.52,-50}, {1.76,0,50}}, {{-1.76,-3.52,-50}, {1.76,0,50}}})"};
        static char const* const expected_trans_strings[] = {
            "0: t=0 -> {}",
            "1: t=0",
            "2: t=2 -> {{0,1.76,0}}",
            "3: t=1 -> {{0,-1.76,0}}",
        };
        static char const* const expected_fill_strings[]
            = {"<UNASSIGNED>", "{u=0, t=1}", "{u=1, t=2}", "m2"};
        static int const expected_volume_nodes[] = {1, 3, 2, 1};
        static char const expected_tree_string[]
            = R"json(["t",["~",0],["S",6],["~",2]])json";

        auto u = this->build_unit(protos, UnivId{1});

        EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
        EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
        EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
        EXPECT_VEC_EQ(expected_bound_strings, bound_strings(u));
        EXPECT_VEC_EQ(expected_trans_strings, transform_strings(u));
        EXPECT_VEC_EQ(expected_fill_strings, fill_strings(u));
        EXPECT_VEC_EQ(expected_volume_nodes, volume_nodes(u));
        EXPECT_JSON_EQ(expected_tree_string, tree_string(u));
    }
    {
        SCOPED_TRACE("ZN1");
        auto u = this->build_unit(protos, UnivId{2});

        EXPECT_JSON_EQ(R"json({"py":10})json", count_surface_types(u));
    }
    {
        SCOPED_TRACE("ZNST");
        auto u = this->build_unit(protos, UnivId{4});

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
        auto u = this->build_unit(protos, UnivId{5});
        static char const* const expected_surface_strings[]
            = {"Cyl z: r=0.01825"};
        static char const* const expected_volume_strings[] = {"F", "-6", "+6"};

        EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
        EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
        EXPECT_EQ(GeoMatId{}, u.background);
    }
}

TEST_F(ZnenvTest, explicit_interior)
{
    this->setup_opts(R"json({
"_format": "g4org-options",
"csg_output_file": null,
"explicit_interior_threshold": 0,
"implicit_parent_boundary": false,
"inline_childless": false,
"inline_singletons": "none",
"inline_unions": false,
"logic": "postfix",
"objects_output_file": null,
"org_output_file": null,
"tol": {"abs": 0.0001, "rel": 0.001},
"unit_length": 1.0,
"verbose_structure": false,
"verbose_volumes": false
})json");

    auto global_proto = this->load("znenv");
    ProtoMap protos{*global_proto};

    static std::string const expected_proto_names[] = {
        "World",
        "ZNENV",
        "ZNEU",
        "ZNTX",
        "ZN1",
        "ZNSL",
        "ZNST",
        "ZNG1",
        "ZNG2",
        "ZNG3",
        "ZNG4",
        "ZNF1",
        "ZNF2",
        "ZNF3",
        "ZNF4",
    };
    EXPECT_VEC_EQ(expected_proto_names, get_proto_names(protos));

    ASSERT_EQ(15, protos.size());
    {
        SCOPED_TRACE("global");
        auto u = this->build_unit(protos, UnivId{0});

        static char const* const expected_volume_strings[]
            = {"any(-0, +1, -2, +3, -4, +5)", "all(+6, -7, +8, -9, +10, -11)"};
        EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
        EXPECT_EQ(GeoMatId{3}, u.background);
    }
    {
        SCOPED_TRACE("ZN1");
        auto u = this->build_unit(protos, UnivId{4});

        EXPECT_JSON_EQ(R"json({"px":2,"py":12,"pz":2})json",
                       count_surface_types(u));
    }
    {
        SCOPED_TRACE("ZNG1");
        auto u = this->build_unit(protos, UnivId{7});
        static char const* const expected_surface_strings[] = {
            "Plane: x=-0.3",
            "Plane: x=0.3",
            "Plane: y=-0.3",
            "Plane: y=0.3",
            "Plane: z=-500",
            "Plane: z=500",
            "Cyl z: r=0.1825",
        };
        static char const* const expected_volume_strings[]
            = {"any(-0, +1, -2, +3, -4, +5)", "all(+4, -5, -6)"};
        static char const* const expected_md_strings[] = {
            "",
            "",
            "ZNG10x0@mx",
            "",
            "ZNG10x0@px",
            "ZNG10x0@my",
            "",
            "ZNG10x0@py",
            "ZNF10x0@mz,ZNG10x0@mz",
            "",
            "ZNF10x0@pz,ZNG10x0@pz",
            "",
            "[EXTERIOR]",
            "ZNF10x0@cz",
            "",
            "ZNF10x0",
        };

        EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
        EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
        EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
        EXPECT_EQ(GeoMatId{1}, u.background);
    }
    {
        SCOPED_TRACE("ZNF1");
        auto u = this->build_unit(protos, UnivId{11});

        static char const* const expected_surface_strings[]
            = {"Plane: z=-500", "Plane: z=500", "Cyl z: r=0.1825"};
        static char const* const expected_volume_strings[]
            = {"any(-0, +1, +2)", "all(+0, -1, -2)"};
        static char const* const expected_md_strings[] = {
            "",
            "",
            "ZNF10x0@mz",
            "",
            "ZNF10x0@pz",
            "",
            "ZNF10x0@cz",
            "",
            "[EXTERIOR]",
            "ZNF10x0",
        };
        static char const* const expected_fill_strings[]
            = {"<UNASSIGNED>", "m0"};

        EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
        EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
        EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
        EXPECT_VEC_EQ(expected_fill_strings, fill_strings(u));
        EXPECT_EQ(GeoMatId{}, u.background);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace g4org
}  // namespace celeritas
