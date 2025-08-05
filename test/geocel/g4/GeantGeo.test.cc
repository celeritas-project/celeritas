//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/g4/GeantGeo.test.cc
//---------------------------------------------------------------------------//
#include <regex>
#include <string_view>
#include <G4LogicalVolume.hh>

#include "corecel/Config.hh"

#include "corecel/ScopedLogStorer.hh"
#include "corecel/cont/Span.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/StringUtils.hh"
#include "corecel/sys/Version.hh"
#include "geocel/GeantGeoParams.hh"
#include "geocel/GenericGeoParameterizedTest.hh"
#include "geocel/GeoParamsOutput.hh"
#include "geocel/GeoTests.hh"
#include "geocel/UnitUtils.hh"
#include "geocel/g4/GeantGeoData.hh"
#include "geocel/g4/GeantGeoTrackView.hh"
#include "geocel/rasterize/SafetyImager.hh"

#include "GeantGeoTestBase.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
namespace
{
auto const geant4_version = celeritas::Version::from_string(
    CELERITAS_USE_GEANT4 ? cmake::geant4_version : "0.0.0");

}  // namespace

//---------------------------------------------------------------------------//

class GeantGeoTest : public GeantGeoTestBase
{
  public:
    using SpanStringView = Span<std::string_view const>;

    SPConstGeo build_geometry() final
    {
        ScopedLogStorer scoped_log_{&celeritas::self_logger(),
                                    LogLevel::warning};
        auto result = this->build_geometry_from_basename();
        EXPECT_VEC_EQ(this->expected_log_levels(), scoped_log_.levels())
            << scoped_log_;
        return result;
    }

    virtual SpanStringView expected_log_levels() const { return {}; }
};

//---------------------------------------------------------------------------//
using CmseTest = GenericGeoParameterizedTest<GeantGeoTest, CmseGeoTest>;

TEST_F(CmseTest, model)
{
    this->impl().test_model();
}

TEST_F(CmseTest, trace)
{
    this->impl().test_trace();
}

TEST_F(CmseTest, imager)
{
    SafetyImager write_image{this->geometry()};

    ImageInput inp;
    inp.lower_left = from_cm({-550, 0, -4000});
    inp.upper_right = from_cm({550, 0, 2000});
    inp.rightward = {0.0, 0.0, 1.0};
    inp.vertical_pixels = 8;

    std::string prefix = "g4";

    write_image(ImageParams{inp}, prefix + "-cmse.jsonl");
}

//---------------------------------------------------------------------------//
using CmsEeBackDeeTest
    = GenericGeoParameterizedTest<GeantGeoTest, CmsEeBackDeeGeoTest>;

TEST_F(CmsEeBackDeeTest, accessors)
{
    this->impl().test_accessors();
}

TEST_F(CmsEeBackDeeTest, model)
{
    this->impl().test_model();
}

TEST_F(CmsEeBackDeeTest, trace)
{
    this->impl().test_trace();
}

//---------------------------------------------------------------------------//
using FourLevelsTest
    = GenericGeoParameterizedTest<GeantGeoTest, FourLevelsGeoTest>;

TEST_F(FourLevelsTest, accessors)
{
    this->impl().test_accessors();
}

TEST_F(FourLevelsTest, model)
{
    this->impl().test_model();
}

TEST_F(FourLevelsTest, trace)
{
    this->impl().test_trace();
}

TEST_F(FourLevelsTest, consecutive_compute)
{
    auto geo = this->make_geo_track_view({-9, -10, -10}, {1, 0, 0});
    ASSERT_FALSE(geo.is_outside());
    EXPECT_EQ("Shape2", this->volume_name(geo));
    EXPECT_FALSE(geo.is_on_boundary());

    auto next = geo.find_next_step(from_cm(10.0));
    EXPECT_SOFT_EQ(4.0, to_cm(next.distance));
    EXPECT_SOFT_EQ(4.0, to_cm(geo.find_safety()));

    next = geo.find_next_step(from_cm(10.0));
    EXPECT_SOFT_EQ(4.0, to_cm(next.distance));
    EXPECT_SOFT_EQ(4.0, to_cm(geo.find_safety()));

    // Find safety from a freshly initialized state
    geo = {from_cm({-9, -10, -10}), {1, 0, 0}};
    EXPECT_SOFT_EQ(4.0, to_cm(geo.find_safety()));
}

TEST_F(FourLevelsTest, detailed_track)
{
    {
        SCOPED_TRACE("rightward along corner");
        auto geo = this->make_geo_track_view({-10, -10, -10}, {1, 0, 0});
        ASSERT_FALSE(geo.is_outside());
        EXPECT_EQ("Shape2", this->volume_name(geo));
        EXPECT_FALSE(geo.is_on_boundary());

        // Check for surfaces up to a distance of 4 units away
        auto next = geo.find_next_step(from_cm(4.0));
        EXPECT_SOFT_EQ(4.0, to_cm(next.distance));
        EXPECT_FALSE(next.boundary);
        next = geo.find_next_step(from_cm(4.0));
        EXPECT_SOFT_EQ(4.0, to_cm(next.distance));
        EXPECT_FALSE(next.boundary);
        geo.move_internal(from_cm(3.5));
        EXPECT_FALSE(geo.is_on_boundary());

        // Find one a bit further, then cross it
        next = geo.find_next_step(from_cm(4.0));
        EXPECT_SOFT_EQ(1.5, to_cm(next.distance));
        EXPECT_TRUE(next.boundary);
        geo.move_to_boundary();
        EXPECT_EQ("Shape2", this->volume_name(geo));

        geo.cross_boundary();
        EXPECT_EQ("Shape1", this->volume_name(geo));
        EXPECT_TRUE(geo.is_on_boundary());

        // Find the next boundary and make sure that nearer distances aren't
        // accepted
        next = geo.find_next_step();
        EXPECT_SOFT_EQ(1.0, to_cm(next.distance));
        EXPECT_TRUE(next.boundary);
        EXPECT_TRUE(geo.is_on_boundary());
        next = geo.find_next_step(from_cm(0.5));
        EXPECT_SOFT_EQ(0.5, to_cm(next.distance));
        EXPECT_FALSE(next.boundary);
    }
    {
        SCOPED_TRACE("inside out");
        auto geo = this->make_geo_track_view({-23.5, 6.5, 6.5}, {-1, 0, 0});
        EXPECT_FALSE(geo.is_outside());
        EXPECT_EQ("World", this->volume_name(geo));

        auto next = geo.find_next_step(from_cm(2));
        EXPECT_SOFT_EQ(0.5, to_cm(next.distance));
        EXPECT_TRUE(next.boundary);

        geo.move_to_boundary();
        EXPECT_FALSE(geo.is_outside());
        geo.cross_boundary();
        EXPECT_TRUE(geo.is_outside());
    }
}

TEST_F(FourLevelsTest, reentrant_boundary)
{
    auto geo = this->make_geo_track_view({15.5, 10, 10}, {-1, 0, 0});
    ASSERT_FALSE(geo.is_outside());
    EXPECT_EQ("Shape1", this->volume_name(geo));
    EXPECT_FALSE(geo.is_on_boundary());

    // Check for surfaces: we should hit the outside of the sphere Shape2
    auto next = geo.find_next_step(from_cm(1.0));
    EXPECT_SOFT_EQ(0.5, to_cm(next.distance));
    // Move to the boundary but scatter perpendicularly, away from the sphere
    geo.move_to_boundary();
    EXPECT_TRUE(geo.is_on_boundary());
    geo.set_dir({0, 1, 0});
    EXPECT_TRUE(geo.is_on_boundary());
    EXPECT_EQ("Shape1", this->volume_name(geo));

    // Move a bit internally, then scatter back toward the sphere
    next = geo.find_next_step(from_cm(10.0));
    EXPECT_SOFT_EQ(6, to_cm(next.distance));
    geo.set_dir({-1, 0, 0});
    EXPECT_EQ("Shape1", this->volume_name(geo));

    // Move to the sphere boundary then scatter still into the sphere
    next = geo.find_next_step(from_cm(10.0));
    EXPECT_SOFT_EQ(1e-13, to_cm(next.distance));
    EXPECT_TRUE(next.boundary);
    geo.move_to_boundary();
    EXPECT_TRUE(geo.is_on_boundary());
    geo.set_dir({0, -1, 0});
    EXPECT_TRUE(geo.is_on_boundary());
    geo.cross_boundary();
    EXPECT_EQ("Shape2", this->volume_name(geo));

    EXPECT_TRUE(geo.is_on_boundary());

    // Travel nearly tangent to the right edge of the sphere, then scatter to
    // still outside
    next = geo.find_next_step(from_cm(1.0));
    EXPECT_SOFT_EQ(9.9794624025613538e-07, to_cm(next.distance));
    geo.move_to_boundary();
    EXPECT_TRUE(geo.is_on_boundary());
    geo.set_dir({1, 0, 0});
    EXPECT_TRUE(geo.is_on_boundary());
    geo.cross_boundary();
    EXPECT_EQ("Shape1", this->volume_name(geo));

    EXPECT_TRUE(geo.is_on_boundary());
    next = geo.find_next_step(from_cm(10.0));
}

TEST_F(FourLevelsTest, safety)
{
    auto geo = this->make_geo_track_view();
    std::vector<real_type> safeties;
    std::vector<real_type> lim_safeties;

    for (auto i : range(11))
    {
        real_type r = from_cm(2.0 * i + 0.1);
        geo = {{r, r, r}, {1, 0, 0}};
        if (!geo.is_outside())
        {
            geo.find_next_step();
            safeties.push_back(to_cm(geo.find_safety()));
            lim_safeties.push_back(to_cm(geo.find_safety(from_cm(1.5))));
        }
    }

    static double const expected_safeties[] = {
        2.9,
        0.9,
        0.1,
        1.7549981495186,
        1.7091034656191,
        4.8267949192431,
        1.3626933041054,
        1.9,
        0.1,
        1.1,
        3.1,
    };
    EXPECT_VEC_SOFT_EQ(expected_safeties, safeties);

    static double const expected_lim_safeties[] = {
        2.9,
        0.9,
        0.1,
        1.7549981495186,
        1.7091034656191,
        4.8267949192431,
        1.3626933041054,
        1.9,
        0.1,
        1.1,
        3.1,
    };
    EXPECT_VEC_SOFT_EQ(expected_lim_safeties, lim_safeties);
}

TEST_F(FourLevelsTest, levels)
{
    auto geo = this->make_geo_track_view({10.0, 10.0, 10.0}, {1, 0, 0});
    EXPECT_EQ("World_PV/env1/Shape1/Shape2", this->unique_volume_name(geo));
    geo.find_next_step();
    geo.move_to_boundary();
    geo.cross_boundary();

    EXPECT_EQ("World_PV/env1/Shape1", this->unique_volume_name(geo));
    geo.find_next_step();
    geo.move_to_boundary();
    geo.cross_boundary();

    EXPECT_EQ("World_PV/env1", this->unique_volume_name(geo));
    geo.find_next_step();
    geo.move_to_boundary();
    geo.cross_boundary();

    EXPECT_EQ("World_PV", this->unique_volume_name(geo));
    geo.find_next_step();
    geo.move_to_boundary();
    geo.cross_boundary();

    EXPECT_EQ("[OUTSIDE]", this->unique_volume_name(geo));
}

//---------------------------------------------------------------------------//
using MultiLevelTest
    = GenericGeoParameterizedTest<GeantGeoTest, MultiLevelGeoTest>;

TEST_F(MultiLevelTest, model)
{
    this->impl().test_model();
}

TEST_F(MultiLevelTest, trace)
{
    this->impl().test_trace();
}

TEST_F(MultiLevelTest, level_strings)
{
    using R2 = Array<double, 2>;

    auto const& vol_inst = this->geometry()->volume_instances();
    auto const& vol = this->geometry()->impl_volumes();

    // Include outer world and center sphere
    std::vector<R2> points{R2{-5, 0}, R2{0, 0}};

    // Loop over outer and inner x and y signs
    for (auto signs : range(1 << 4))
    {
        auto get_sign = [signs](int i) {
            CELER_ASSERT(i < 4);
            return signs & (1 << i) ? -1 : 1;
        };
        R2 point{0, 0};
        point[0] += 2.75 * get_sign(0);
        point[1] += 2.75 * get_sign(1);
        point[0] += 10.0 * get_sign(2);
        point[1] += 10.0 * get_sign(3);
        points.push_back(point);
    }

    std::vector<std::string> all_vol;
    std::vector<std::string> all_vol_inst;
    for (R2 xy : points)
    {
        auto geo = this->make_geo_track_view({xy[0], xy[1], 0.0}, {1, 0, 0});

        auto level = geo.level();
        CELER_ASSERT(level && level >= LevelId{0});
        std::vector<VolumeInstanceId> inst_ids(level.get() + 1);
        geo.volume_instance_id(make_span(inst_ids));
        std::vector<std::string> names(inst_ids.size());
        for (auto i : range(inst_ids.size()))
        {
            names[i] = to_string(vol_inst.at(inst_ids[i]));
        }
        all_vol_inst.push_back(to_string(repr(names)));
        all_vol.push_back(to_string(vol.at(geo.impl_volume_id())));
    }

    static char const* const expected_all_vol_inst[] = {
        "{\"world_PV\"}",
        R"({"world_PV", "topsph1"})",
        R"({"world_PV", "topbox1", "boxsph1@0"})",
        R"({"world_PV", "topbox1"})",
        R"({"world_PV", "topbox1", "boxtri@0"})",
        R"({"world_PV", "topbox1", "boxsph2@0"})",
        R"({"world_PV", "topbox2", "boxsph1@0"})",
        R"({"world_PV", "topbox2"})",
        R"({"world_PV", "topbox2", "boxtri@0"})",
        R"({"world_PV", "topbox2", "boxsph2@0"})",
        R"({"world_PV", "topbox4", "boxtri@1"})",
        R"({"world_PV", "topbox4", "boxsph2@1"})",
        R"({"world_PV", "topbox4", "boxsph1@1"})",
        R"({"world_PV", "topbox4"})",
        R"({"world_PV", "topbox3"})",
        R"({"world_PV", "topbox3", "boxsph2@0"})",
        R"({"world_PV", "topbox3", "boxsph1@0"})",
        R"({"world_PV", "topbox3", "boxtri@0"})",
    };
    static char const* const expected_all_vol[] = {
        "world",
        "sph",
        "sph",
        "box",
        "tri",
        "sph",
        "sph",
        "box",
        "tri",
        "sph",
        "tri_refl",
        "sph_refl",
        "sph_refl",
        "box_refl",
        "box",
        "sph",
        "sph",
        "tri",
    };

    EXPECT_VEC_EQ(expected_all_vol_inst, all_vol_inst);
    EXPECT_VEC_EQ(expected_all_vol, all_vol);
}

//---------------------------------------------------------------------------//
using OpticalSurfacesTest
    = GenericGeoParameterizedTest<GeantGeoTest, OpticalSurfacesGeoTest>;

TEST_F(OpticalSurfacesTest, model)
{
    this->impl().test_model();
}

TEST_F(OpticalSurfacesTest, trace)
{
    this->impl().test_trace();
}

//---------------------------------------------------------------------------//
class PincellTest : public GeantGeoTest
{
    std::string geometry_basename() const override { return "pincell"; }
};

TEST_F(PincellTest, imager)
{
    SafetyImager write_image{this->geometry()};

    ImageInput inp;
    inp.lower_left = from_cm({-12, -12, 0});
    inp.upper_right = from_cm({12, 12, 0});
    inp.rightward = {1.0, 0.0, 0.0};
    inp.vertical_pixels = 8;

    write_image(ImageParams{inp}, "g4-pincell-xy-mid.jsonl");

    inp.lower_left[2] = inp.upper_right[2] = from_cm(-5.5);
    write_image(ImageParams{inp}, "g4-pincell-xy-lo.jsonl");

    inp.lower_left = from_cm({-12, 0, -12});
    inp.upper_right = from_cm({12, 0, 12});
    write_image(ImageParams{inp}, "g4-pincell-xz-mid.jsonl");
}

//---------------------------------------------------------------------------//

using PolyhedraTest
    = GenericGeoParameterizedTest<GeantGeoTest, PolyhedraGeoTest>;

TEST_F(PolyhedraTest, model)
{
    this->impl().test_model();
}

TEST_F(PolyhedraTest, trace)
{
    TestImpl(this).test_trace();
}

//---------------------------------------------------------------------------//
using ReplicaTest = GenericGeoParameterizedTest<GeantGeoTest, ReplicaGeoTest>;

TEST_F(ReplicaTest, model)
{
    this->impl().test_model();
}

TEST_F(ReplicaTest, trace)
{
    this->impl().test_trace();
}

TEST_F(ReplicaTest, volume_stack)
{
    this->impl().test_volume_stack();
}

TEST_F(ReplicaTest, level_strings)
{
    using R2 = Array<double, 2>;

    auto const& geo_params = *this->geometry();
    auto const& vol_inst = geo_params.volume_instances();

    static R2 const points[] = {
        {-435, 550},
        {-460, 550},
        {-400, 650},
        {-450, 650},
        {-450, 700},
    };

    std::vector<std::string> all_vol_inst;
    for (R2 xz : points)
    {
        auto geo = this->make_geo_track_view({xz[0], 0.0, xz[1]}, {1, 0, 0});

        auto level = geo.level();
        CELER_ASSERT(level && level >= LevelId{0});
        std::vector<VolumeInstanceId> inst_ids(level.get() + 1);
        geo.volume_instance_id(make_span(inst_ids));
        std::vector<std::string> names(inst_ids.size());
        for (auto i : range(inst_ids.size()))
        {
            Label lab = vol_inst.at(inst_ids[i]);
            if (auto phys_inst = geo_params.id_to_geant(inst_ids[i]))
            {
                if (phys_inst.replica)
                {
                    lab.ext += '+';
                    lab.ext += std::to_string(phys_inst.replica.get());
                }
            }
            names[i] = to_string(lab);
        }
        all_vol_inst.push_back(to_string(repr(names)));
    }

    static char const* const expected_all_vol_inst[] = {
        R"({"world_PV", "fSecondArmPhys", "EMcalorimeter", "cell_param@+14"})",
        R"({"world_PV", "fSecondArmPhys", "EMcalorimeter", "cell_param@+6"})",
        R"({"world_PV", "fSecondArmPhys", "HadCalorimeter", "HadCalColumn_PV@+4", "HadCalCell_PV@+1", "HadCalLayer_PV@+2"})",
        R"({"world_PV", "fSecondArmPhys", "HadCalorimeter", "HadCalColumn_PV@+2", "HadCalCell_PV@+1", "HadCalLayer_PV@+7"})",
        R"({"world_PV", "fSecondArmPhys", "HadCalorimeter", "HadCalColumn_PV@+3", "HadCalCell_PV@+1", "HadCalLayer_PV@+16"})",
    };

    EXPECT_VEC_EQ(expected_all_vol_inst, all_vol_inst);
}

//---------------------------------------------------------------------------//

using SimpleCmsTest
    = GenericGeoParameterizedTest<GeantGeoTest, SimpleCmsGeoTest>;

TEST_F(SimpleCmsTest, model)
{
    this->impl().test_model();
}

TEST_F(SimpleCmsTest, trace)
{
    this->impl().test_trace();
}

TEST_F(SimpleCmsTest, detailed_track)
{
    // Templated test
    SimpleCmsGeoTest::test_detailed_tracking(this);
}

//---------------------------------------------------------------------------//
class SolidsTest
    : public GenericGeoParameterizedTest<GeantGeoTest, SolidsGeoTest>
{
    SpanStringView expected_log_levels() const final
    {
        if (geant4_version < Version{11})
            return {};

        static std::string_view const levels[] = {"error"};
        return make_span(levels);
    }
};

//---------------------------------------------------------------------------//
TEST_F(SolidsTest, output)
{
    GeoParamsOutput out(this->geometry());
    EXPECT_EQ("geometry", out.label());

    if (CELERITAS_UNITS == CELERITAS_UNITS_CGS)
    {
        auto actual = to_string(out);
        std::regex pattern(R"("",)");
        actual = std::regex_replace(actual, pattern, "");
        EXPECT_JSON_EQ(
            R"json({"_category":"internal","_label":"geometry","bbox":[[-600.0,-300.0,-75.0],[600.0,300.0,75.0]],"supports_safety":true,"volumes":{"label":["box500","cone1","para1","sphere1","parabol1","trap1","trd1","trd2","trd3_refl@1","tube100","boolean1","polycone1","genPocone1","ellipsoid1","tetrah1","orb1","polyhedr1","hype1","elltube1","ellcone1","arb8b","arb8a","xtru1","World","trd3_refl@0"]}})json",
            actual);
    }
}
//---------------------------------------------------------------------------//

TEST_F(SolidsTest, accessors)
{
    TestImpl(this).test_accessors();
}

//---------------------------------------------------------------------------//

TEST_F(SolidsTest, trace)
{
    TestImpl(this).test_trace();
}

//---------------------------------------------------------------------------//

TEST_F(SolidsTest, reflected_vol)
{
    auto geo = this->make_geo_track_view({-500, -125, 0}, {0, 1, 0});
    EXPECT_EQ(25, geo.impl_volume_id().unchecked_get());
    auto const& label
        = this->geometry()->impl_volumes().at(geo.impl_volume_id());
    EXPECT_EQ("trd3_refl", label.name);
    EXPECT_FALSE(ends_with(label.ext, "_refl"));
}

TEST_F(SolidsTest, DISABLED_imager)
{
    SafetyImager write_image{this->geometry()};

    ImageInput inp;
    inp.lower_left = from_cm({-550, -250, 5});
    inp.upper_right = from_cm({550, 250, 5});
    inp.rightward = {1.0, 0.0, 0.0};
    inp.vertical_pixels = 8;

    write_image(ImageParams{inp}, "g4-solids-xy-hi.jsonl");

    inp.lower_left[2] = inp.upper_right[2] = from_cm(-5);
    write_image(ImageParams{inp}, "g4-solids-xy-lo.jsonl");
}

//---------------------------------------------------------------------------//
using TilecalPlugTest
    = GenericGeoParameterizedTest<GeantGeoTest, TilecalPlugGeoTest>;

TEST_F(TilecalPlugTest, model)
{
    this->impl().test_model();
}

TEST_F(TilecalPlugTest, trace)
{
    this->impl().test_trace();
}

//---------------------------------------------------------------------------//
using TransformedBoxTest
    = GenericGeoParameterizedTest<GeantGeoTest, TransformedBoxGeoTest>;

TEST_F(TransformedBoxTest, accessors)
{
    this->impl().test_accessors();
}

TEST_F(TransformedBoxTest, model)
{
    this->impl().test_model();
}

TEST_F(TransformedBoxTest, trace)
{
    this->impl().test_trace();
}

//---------------------------------------------------------------------------//

class TwoBoxesTest
    : public GenericGeoParameterizedTest<GeantGeoTest, TwoBoxesGeoTest>
{
};

TEST_F(TwoBoxesTest, accessors)
{
    this->impl().test_accessors();
}

TEST_F(TwoBoxesTest, model)
{
    this->impl().test_model();
}

TEST_F(TwoBoxesTest, track)
{
    // Templated test
    TwoBoxesGeoTest::test_detailed_tracking(this);
}

//---------------------------------------------------------------------------//
using ZnenvTest = GenericGeoParameterizedTest<GeantGeoTest, ZnenvGeoTest>;

TEST_F(ZnenvTest, model)
{
    this->impl().test_model();
}

TEST_F(ZnenvTest, trace)
{
    this->impl().test_trace();
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
