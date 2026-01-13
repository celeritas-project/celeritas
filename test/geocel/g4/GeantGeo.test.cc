//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/g4/GeantGeo.test.cc
//---------------------------------------------------------------------------//
#include <regex>
#include <string_view>
#include <G4LogicalVolume.hh>
#include <G4StateManager.hh>
#include <G4VSensitiveDetector.hh>

#include "corecel/Config.hh"

#include "corecel/ScopedLogStorer.hh"
#include "corecel/cont/Span.hh"
#include "corecel/io/ColorUtils.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/StringUtils.hh"
#include "corecel/sys/Version.hh"
#include "geocel/GeantGeoParams.hh"
#include "geocel/GenericGeoParameterizedTest.hh"
#include "geocel/GenericGeoResults.hh"
#include "geocel/GeoParamsOutput.hh"
#include "geocel/GeoTests.hh"
#include "geocel/ScopedGeantExceptionHandler.hh"
#include "geocel/ScopedGeantLogger.hh"
#include "geocel/UnitUtils.hh"
#include "geocel/VolumeParams.hh"  // IWYU pragma: keep
#include "geocel/inp/Model.hh"  // IWYU pragma: keep
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

    static void SetUpTestSuite()
    {
        // Print version number for verification on CI systems etc.
        static bool const have_printed_ = [] {
            using namespace celeritas::cmake;
            cout << color_code('x') << "Using Geant4 v" << geant4_version
                 << " (" << geant4_options << ")" << color_code(' ') << endl;
            return true;
        }();
        EXPECT_TRUE(have_printed_);
    }

    SPConstGeantGeo build_geant_geo(std::string const& filename) const final
    {
        ScopedLogStorer scoped_log_{&celeritas::world_logger(),
                                    LogLevel::warning};
        auto result = GeantGeoParams::from_gdml(filename);
        EXPECT_VEC_EQ(this->expected_log_levels(), scoped_log_.levels())
            << scoped_log_;
        return result;
    }

    GenericGeoModelInp summarize_model()
    {
        return GenericGeoModelInp::from_model_input(
            this->geometry()->make_model_input());
    }

    virtual SpanStringView expected_log_levels() const { return {}; }

    ScopedGeantExceptionHandler exception_handler;
    ScopedGeantLogger logger{celeritas::world_logger()};

    void SetUp() override
    {
        GeantGeoTestBase::SetUp();
        ASSERT_TRUE(this->geometry());

        auto* sm = G4StateManager::GetStateManager();
        CELER_ASSERT(sm);
        // Have ScopedGeantExceptionHandler treat tracking errors like runtime
        EXPECT_TRUE(sm->SetNewState(G4ApplicationState::G4State_EventProc));
    }

    void TearDown() override
    {
        ASSERT_TRUE(this->geometry());

        // Restore G4 state just in case it matters
        auto* sm = G4StateManager::GetStateManager();
        EXPECT_TRUE(sm->SetNewState(G4ApplicationState::G4State_PreInit));
        GeantGeoTestBase::TearDown();
    }
};

//---------------------------------------------------------------------------//
using AtlasHgtdTest
    = GenericGeoParameterizedTest<GeantGeoTest, AtlasHgtdGeoTest>;

TEST_F(AtlasHgtdTest, model)
{
    auto result = this->summarize_model();
    GenericGeoModelInp ref;
    ref.volume.labels = {"SPlate", "HGTD", "ITK", "Atlas"};
    ref.volume.materials = {0, 1, 1, 1};
    ref.volume.daughters = {{}, {3, 4, 5, 6, 7, 8, 9, 10}, {2}, {1}};
    ref.volume_instance.labels = {
        "Atlas_PV",
        "ITK",
        "HGTD",
        "SPlate_4@0",
        "SPlate_5@0",
        "SPlate_6@0",
        "SPlate_7@0",
        "SPlate_4@1",
        "SPlate_5@1",
        "SPlate_6@1",
        "SPlate_7@1",
    };
    ref.volume_instance.volumes = {
        3,
        2,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    };
    ref.world = "Atlas";
    EXPECT_REF_EQ(ref, result);
}

TEST_F(AtlasHgtdTest, trace)
{
    this->impl().test_trace();
}

TEST_F(AtlasHgtdTest, volume_stack)
{
    this->impl().test_volume_stack();
}

TEST_F(AtlasHgtdTest, detailed_track)
{
    this->impl().test_detailed_tracking();
}

//---------------------------------------------------------------------------//
using CmseTest = GenericGeoParameterizedTest<GeantGeoTest, CmseGeoTest>;

TEST_F(CmseTest, model)
{
    auto result = this->summarize_model();
    GenericGeoModelInp ref;
    ref.volume.labels = {
        "CMStoZDC", "ZDCtoFP420", "Tracker", "CALO",    "MUON",
        "BEAM",     "BEAM1",      "BEAM2",   "BEAM3",   "TrackerPixelNose",
        "VCAL",     "TotemT1",    "TotemT2", "CastorF", "CastorB",
        "OQUA",     "BSC2",       "ZDC",     "CMSE",    "OCMS",
    };
    ref.volume.materials = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    };
    ref.volume.daughters = {
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {
            2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
            18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
        },
        {1},
    };
    ref.volume_instance.labels = {
        "OCMS_PV",
        "CMSE",
        "CMStoZDC@0",
        "CMStoZDC@1",
        "ZDCtoFP420@0",
        "ZDCtoFP420@1",
        "Tracker",
        "CALO",
        "MUON",
        "BEAM@0",
        "BEAM@1",
        "BEAM1@0",
        "BEAM1@1",
        "BEAM2@0",
        "BEAM2@1",
        "BEAM3@0",
        "BEAM3@1",
        "TrackerPixelNose@0",
        "TrackerPixelNose@1",
        "VCAL@0",
        "VCAL@1",
        "TotemT1@0",
        "TotemT1@1",
        "TotemT2@0",
        "TotemT2@1",
        "CastorF",
        "CastorB",
        "OQUA@0",
        "OQUA@1",
        "BSC2@0",
        "BSC2@1",
        "ZDC@0",
        "ZDC@1",
    };
    ref.volume_instance.volumes = {
        19, 18, 0,  0,  1,  1,  2,  3,  4,  5,  5,  6,  6,  7,  7,  8,  8,
        9,  9,  10, 10, 11, 11, 12, 12, 13, 14, 15, 15, 16, 16, 17, 17,
    };
    ref.world = "OCMS";
    EXPECT_REF_EQ(ref, result);
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
    auto result = this->summarize_model();
    GenericGeoModelInp ref;
    ref.volume.labels = {
        "EEBackPlate",
        "EESRing",
        "EEBackQuad",
        "EEBackDee",
        "EEBackQuad_refl",
        "EEBackPlate_refl",
        "EESRing_refl",
    };
    ref.volume.materials = {0, 0, 1, 1, 1, 0, 0};
    ref.volume.daughters = {{}, {}, {2, 3}, {1, 4}, {5, 6}, {}, {}};
    ref.volume_instance.labels = {
        "EEBackDee_PV",
        "EEBackQuad@0",
        "EEBackPlate@0",
        "EESRing@0",
        "EEBackQuad@1",
        "EEBackPlate@1",
        "EESRing@1",
    };
    ref.volume_instance.volumes = {3, 2, 0, 1, 4, 5, 6};
    ref.world = "EEBackDee";
    ref.detector.labels = {"ee_back_plate", "ee_s_ring"};
    ref.detector.volumes = {{0, 5}, {1, 6}};
    EXPECT_REF_EQ(ref, result);
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
    auto result = this->summarize_model();
    GenericGeoModelInp ref;
    ref.volume.labels = {"Shape2", "Shape1", "Envelope", "World"};
    ref.volume.materials = {0, 1, 2, 3};
    ref.volume.daughters = {{}, {3}, {2}, {1, 4, 5, 6, 7, 8, 9, 10}};
    ref.volume_instance.labels = {
        "World_PV",
        "env1",
        "Shape1",
        "Shape2",
        "env2",
        "env3",
        "env4",
        "env5",
        "env6",
        "env7",
        "env8",
    };
    ref.volume_instance.volumes = {
        3,
        2,
        1,
        0,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
    };
    ref.world = "World";
    ref.region.labels = {"envelope_region"};
    ref.region.volumes = {{0, 1, 2}};
    EXPECT_REF_EQ(ref, result);
}

TEST_F(FourLevelsTest, trace)
{
    this->impl().test_trace();
}

TEST_F(FourLevelsTest, consecutive_compute)
{
    this->impl().test_consecutive_compute();
}

TEST_F(FourLevelsTest, detailed_track)
{
    ScopedLogStorer scoped_log_{&self_logger()};
    this->impl().test_detailed_tracking();

    // "Finding next step up to ... when previous step 4 was already
    // calculated"
    static char const* const expected_log_levels[] = {"warning", "warning"};
    EXPECT_VEC_EQ(expected_log_levels, scoped_log_.levels()) << scoped_log_;
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
using LarSphereTest
    = GenericGeoParameterizedTest<GeantGeoTest, LarSphereGeoTest>;

TEST_F(LarSphereTest, model)
{
    auto result = this->summarize_model();

    GenericGeoModelInp ref;
    ref.volume.labels
        = {"sphere", "detshell_top", "detshell_bot", "detshell", "world"};
    ref.volume.materials = {1, 1, 1, 0, 0};
    ref.volume.daughters = {{}, {}, {}, {2, 3}, {1, 4}};
    ref.volume_instance.labels = {
        "world_PV",
        "detshell_PV",
        "detshell_top_PV",
        "detshell_bot_PV",
        "sphere_PV",
    };
    ref.volume_instance.volumes = {4, 3, 1, 2, 0};
    ref.world = "world";
    ref.detector.labels = {"detshell"};
    ref.detector.volumes = {{1, 2}};
    EXPECT_REF_EQ(ref, result);
}

TEST_F(LarSphereTest, trace)
{
    this->impl().test_trace();
}

TEST_F(LarSphereTest, volume_stack)
{
    this->impl().test_volume_stack();
}

//---------------------------------------------------------------------------//
using MultiLevelTest
    = GenericGeoParameterizedTest<GeantGeoTest, MultiLevelGeoTest>;

TEST_F(MultiLevelTest, model)
{
    auto result = this->summarize_model();
    GenericGeoModelInp ref;
    ref.volume.labels
        = {"sph", "tri", "box", "world", "box_refl", "sph_refl", "tri_refl"};
    ref.volume.materials = {0, 0, 1, 0, 1, 0, 0};
    ref.volume.daughters
        = {{}, {}, {2, 3, 4}, {1, 5, 6, 7, 8}, {9, 10, 11}, {}, {}};
    ref.volume_instance.labels = {
        "world_PV",
        "topbox1",
        "boxsph1@0",
        "boxsph2@0",
        "boxtri@0",
        "topsph1",
        "topbox2",
        "topbox3",
        "topbox4",
        "boxsph1@1",
        "boxsph2@1",
        "boxtri@1",
    };
    ref.volume_instance.volumes = {
        3,
        2,
        0,
        0,
        1,
        0,
        2,
        2,
        4,
        5,
        5,
        6,
    };
    ref.world = "world";
    ref.detector.labels = {"sph_sd"};
    ref.detector.volumes = {{0, 5}};
    ref.region.labels = {"sph_region", "tri_region", "box_region"};
    ref.region.volumes = {{0, 5}, {1, 6}, {2, 4}};
    EXPECT_REF_EQ(ref, result);
}

TEST_F(MultiLevelTest, sd_creation)
{
    auto const& geo = *this->geometry();

    auto* lv = geo.id_to_geant(VolumeId{0});  // sph; see model above
    auto* sd = lv->GetSensitiveDetector();
    ASSERT_TRUE(sd);
    EXPECT_EQ("sph_sd", sd->GetName());

    auto* lv2 = geo.id_to_geant(VolumeId{5});  // sph_refl
    auto* sd2 = lv2->GetSensitiveDetector();
    ASSERT_TRUE(sd2);
    EXPECT_EQ(sd, sd2);
}

TEST_F(MultiLevelTest, trace)
{
    this->impl().test_trace();
}

TEST_F(MultiLevelTest, volume_level)
{
    this->impl().test_volume_level();
}

TEST_F(MultiLevelTest, volume_stack)
{
    this->impl().test_volume_stack();
}

//---------------------------------------------------------------------------//
using OpticalSurfacesTest
    = GenericGeoParameterizedTest<GeantGeoTest, OpticalSurfacesGeoTest>;

TEST_F(OpticalSurfacesTest, model)
{
    auto result = this->summarize_model();
    GenericGeoModelInp ref;
    ref.volume.labels = {"lar_sphere", "death", "tube1_mid", "tube2", "world"};
    ref.volume.materials = {1, 2, 2, 2, 3};
    ref.volume.daughters = {{}, {}, {}, {}, {1, 2, 3, 4, 5}};
    ref.volume_instance.labels = {
        "world_PV",
        "lar_pv",
        "death_pv",
        "tube2_below_pv",
        "tube1_mid_pv",
        "tube2_above_pv",
    };
    ref.volume_instance.volumes = {4, 0, 1, 3, 2, 3};
    ref.world = "world";
    ref.surface.labels = {
        "sphere_skin",
        "tube2_skin",
        "below_to_1",
        "mid_to_below",
        "mid_to_above",
    };
    ref.surface.volumes = {"0", "3", "3->4", "4->3", "4->5"};
    EXPECT_REF_EQ(ref, result);
}

TEST_F(OpticalSurfacesTest, trace)
{
    this->impl().test_trace();
}

//---------------------------------------------------------------------------//
class PincellTest : public GeantGeoTest
{
    std::string_view gdml_basename() const override { return "pincell"; }
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
    auto result = this->summarize_model();
    GenericGeoModelInp ref;
    ref.volume.labels = {
        "tri",
        "tri_third",
        "tri_half",
        "tri_full",
        "quad",
        "quad_third",
        "quad_half",
        "quad_full",
        "penta",
        "penta_third",
        "penta_half",
        "penta_full",
        "hex",
        "hex_third",
        "hex_half",
        "hex_full",
        "world",
    };
    ref.volume.materials = {
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
    };
    ref.volume.daughters = {
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
        },
    };
    ref.volume_instance.labels = {
        "world_PV",
        "tri0_pv",
        "tri30_pv",
        "tri60_pv",
        "tri90_pv",
        "quad0_pv",
        "quad30_pv",
        "quad60_pv",
        "quad90_pv",
        "penta0_pv",
        "penta30_pv",
        "penta60_pv",
        "penta90_pv",
        "hex0_pv",
        "hex30_pv",
        "hex60_pv",
        "hex90_pv",
    };
    ref.volume_instance.volumes = {
        16,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
    };
    ref.world = "world";
    EXPECT_REF_EQ(ref, result);
}

TEST_F(PolyhedraTest, trace)
{
    TestImpl(this).test_trace();
}

//---------------------------------------------------------------------------//
class ReplicaTest
    : public GenericGeoParameterizedTest<GeantGeoTest, ReplicaGeoTest>
{
  public:
    //! Half-distance is off by ~2e-12
    GenericGeoTrackingTolerance tracking_tol() const override
    {
        auto result = GeantGeoTest::tracking_tol();
        result.distance = 1e-11;
        return result;
    }
};

TEST_F(ReplicaTest, model)
{
    auto result = this->summarize_model();
    // clang-format off
    GenericGeoModelInp ref;
    ref.volume.labels = {"magnetic", "hodoscope1", "wirePlane1", "chamber1", "firstArm", "hodoscope2", "wirePlane2", "chamber2", "cell", "EMcalorimeter", "HadCalScinti", "HadCalLayer", "HadCalCell", "HadCalColumn", "HadCalorimeter", "secondArm", "world",};
    ref.volume.materials = {0, 1, 2, 2, 0, 1, 2, 2, 3, 3, 1, 4, 4, 4, 4, 0, 0,};
    ref.volume.daughters = {{}, {}, {}, {19}, {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23,}, {}, {}, {51}, {}, {57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136,}, {}, {170}, {150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169,}, {148, 149}, {138, 139, 140, 141, 142, 143, 144, 145, 146, 147,}, {25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 52, 53, 54, 55, 56, 137,}, {1, 2, 24},};
    ref.volume_instance.labels = {"world_PV", "magnetic", "firstArm", "hodoscope1@0", "hodoscope1@1", "hodoscope1@2", "hodoscope1@3", "hodoscope1@4", "hodoscope1@5", "hodoscope1@6", "hodoscope1@7", "hodoscope1@8", "hodoscope1@9", "hodoscope1@10", "hodoscope1@11", "hodoscope1@12", "hodoscope1@13", "hodoscope1@14", "chamber1@0", "wirePlane1", "chamber1@1", "chamber1@2", "chamber1@3", "chamber1@4", "fSecondArmPhys", "hodoscope2@0", "hodoscope2@1", "hodoscope2@2", "hodoscope2@3", "hodoscope2@4", "hodoscope2@5", "hodoscope2@6", "hodoscope2@7", "hodoscope2@8", "hodoscope2@9", "hodoscope2@10", "hodoscope2@11", "hodoscope2@12", "hodoscope2@13", "hodoscope2@14", "hodoscope2@15", "hodoscope2@16", "hodoscope2@17", "hodoscope2@18", "hodoscope2@19", "hodoscope2@20", "hodoscope2@21", "hodoscope2@22", "hodoscope2@23", "hodoscope2@24", "chamber2@0", "wirePlane2", "chamber2@1", "chamber2@2", "chamber2@3", "chamber2@4", "EMcalorimeter", "cell_param@0", "cell_param@1", "cell_param@2", "cell_param@3", "cell_param@4", "cell_param@5", "cell_param@6", "cell_param@7", "cell_param@8", "cell_param@9", "cell_param@10", "cell_param@11", "cell_param@12", "cell_param@13", "cell_param@14", "cell_param@15", "cell_param@16", "cell_param@17", "cell_param@18", "cell_param@19", "cell_param@20", "cell_param@21", "cell_param@22", "cell_param@23", "cell_param@24", "cell_param@25", "cell_param@26", "cell_param@27", "cell_param@28", "cell_param@29", "cell_param@30", "cell_param@31", "cell_param@32", "cell_param@33", "cell_param@34", "cell_param@35", "cell_param@36", "cell_param@37", "cell_param@38", "cell_param@39", "cell_param@40", "cell_param@41", "cell_param@42", "cell_param@43", "cell_param@44", "cell_param@45", "cell_param@46", "cell_param@47", "cell_param@48", "cell_param@49", "cell_param@50", "cell_param@51", "cell_param@52", "cell_param@53", "cell_param@54", "cell_param@55", "cell_param@56", "cell_param@57", "cell_param@58", "cell_param@59", "cell_param@60", "cell_param@61", "cell_param@62", "cell_param@63", "cell_param@64", "cell_param@65", "cell_param@66", "cell_param@67", "cell_param@68", "cell_param@69", "cell_param@70", "cell_param@71", "cell_param@72", "cell_param@73", "cell_param@74", "cell_param@75", "cell_param@76", "cell_param@77", "cell_param@78", "cell_param@79", "HadCalorimeter", "HadCalColumn_PV@0", "HadCalColumn_PV@1", "HadCalColumn_PV@2", "HadCalColumn_PV@3", "HadCalColumn_PV@4", "HadCalColumn_PV@5", "HadCalColumn_PV@6", "HadCalColumn_PV@7", "HadCalColumn_PV@8", "HadCalColumn_PV@9", "HadCalCell_PV@0", "HadCalCell_PV@1", "HadCalLayer_PV@0", "HadCalLayer_PV@1", "HadCalLayer_PV@2", "HadCalLayer_PV@3", "HadCalLayer_PV@4", "HadCalLayer_PV@5", "HadCalLayer_PV@6", "HadCalLayer_PV@7", "HadCalLayer_PV@8", "HadCalLayer_PV@9", "HadCalLayer_PV@10", "HadCalLayer_PV@11", "HadCalLayer_PV@12", "HadCalLayer_PV@13", "HadCalLayer_PV@14", "HadCalLayer_PV@15", "HadCalLayer_PV@16", "HadCalLayer_PV@17", "HadCalLayer_PV@18", "HadCalLayer_PV@19", "HadCalScinti",};
    ref.volume_instance.volumes = {16, 0, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 2, 3, 3, 3, 3, 15, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 7, 6, 7, 7, 7, 7, 9, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 14, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 12, 12, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 10,};
    ref.world = "world";
    EXPECT_REF_EQ(ref, result);
    // clang-format on
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

    auto const& vol_inst = this->volumes()->volume_instance_labels();

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

        auto depth = geo.volume_level();
        CELER_ASSERT(depth && depth >= VolumeLevelId{0});
        std::vector<VolumeInstanceId> inst_ids(depth.get() + 1);
        geo.volume_instance_id(make_span(inst_ids));
        std::vector<std::string> names(inst_ids.size());
        for (auto i : range(inst_ids.size()))
        {
            Label lab = vol_inst.at(inst_ids[i]);
            names[i] = to_string(lab);
        }
        all_vol_inst.push_back(to_string(repr(names)));
    }

    static char const* const expected_all_vol_inst[] = {
        R"({"world_PV", "fSecondArmPhys", "EMcalorimeter", "cell_param@14"})",
        R"({"world_PV", "fSecondArmPhys", "EMcalorimeter", "cell_param@6"})",
        R"({"world_PV", "fSecondArmPhys", "HadCalorimeter", "HadCalColumn_PV@4", "HadCalCell_PV@1", "HadCalLayer_PV@2"})",
        R"({"world_PV", "fSecondArmPhys", "HadCalorimeter", "HadCalColumn_PV@2", "HadCalCell_PV@1", "HadCalLayer_PV@7"})",
        R"({"world_PV", "fSecondArmPhys", "HadCalorimeter", "HadCalColumn_PV@3", "HadCalCell_PV@1", "HadCalLayer_PV@16"})",
    };

    EXPECT_VEC_EQ(expected_all_vol_inst, all_vol_inst);
}

//---------------------------------------------------------------------------//

using SimpleCmsTest
    = GenericGeoParameterizedTest<GeantGeoTest, SimpleCmsGeoTest>;

TEST_F(SimpleCmsTest, model)
{
    auto result = this->summarize_model();
    GenericGeoModelInp ref;
    ref.volume.labels = {"vacuum_tube",
                         "si_tracker",
                         "em_calorimeter",
                         "had_calorimeter",
                         "sc_solenoid",
                         "fe_muon_chambers",
                         "world"};
    ref.volume.materials = {0, 1, 2, 3, 4, 5, 0};
    ref.volume.daughters = {{}, {}, {}, {}, {}, {}, {1, 2, 3, 4, 5, 6}};
    ref.volume_instance.labels = {"world_PV",
                                  "vacuum_tube_pv",
                                  "si_tracker_pv",
                                  "em_calorimeter_pv",
                                  "had_calorimeter_pv",
                                  "sc_solenoid_pv",
                                  "iron_muon_chambers_pv"};
    ref.volume_instance.volumes = {6, 0, 1, 2, 3, 4, 5};
    ref.world = "world";
    ref.detector.labels = {"si_tracker_sd", "em_calorimeter_sd"};
    ref.detector.volumes = {{1}, {2}};
    EXPECT_REF_EQ(ref, result);
}

TEST_F(SimpleCmsTest, trace)
{
    this->impl().test_trace();
}

TEST_F(SimpleCmsTest, detailed_track)
{
    ScopedLogStorer scoped_log_{&self_logger()};
    this->impl().test_detailed_tracking();
    if (geant4_version >= Version{11})
    {
        // G4 11.3: "Accuracy error or slightly inaccurate position shift."
        static char const* const expected_log_levels[] = {"error", "error"};
        EXPECT_VEC_EQ(expected_log_levels, scoped_log_.levels()) << scoped_log_;
    }
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
    ScopedLogStorer scoped_log_{&self_logger()};
    TestImpl(this).test_trace();
    if (geant4_version >= Version{11})
    {
        // G4 11.3 report normal directions perpendicular to track direction
        // due to coincident surfaces; and at least one of the tangents is
        // machine-dependent
        EXPECT_GE(scoped_log_.levels().size(), 3) << scoped_log_;
    }
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
    auto result = this->summarize_model();
    GenericGeoModelInp ref;
    ref.volume.labels = {"Tile_Absorber", "Tile_Plug1Module", "Tile_ITCModule"};
    ref.volume.materials = {0, 1, 1};
    ref.volume.daughters = {{}, {2}, {1}};
    ref.volume_instance.labels
        = {"Tile_ITCModule_PV", "Tile_Plug1Module", "Tile_Absorber"};
    ref.volume_instance.volumes = {2, 1, 0};
    ref.world = "Tile_ITCModule";
    EXPECT_REF_EQ(ref, result);
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
    auto result = this->summarize_model();
    GenericGeoModelInp ref;
    ref.volume.labels = {"simple", "tiny", "enclosing", "world"};
    ref.volume.materials = {0, 0, 0, 0};
    ref.volume.daughters = {{}, {}, {3}, {1, 2, 4}};
    ref.volume_instance.labels
        = {"world_PV", "transrot", "default", "rot", "trans"};
    ref.volume_instance.volumes = {3, 0, 2, 1, 0};
    ref.world = "world";
    EXPECT_REF_EQ(ref, result);
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

TEST_F(TwoBoxesTest, detailed_tracking)
{
    this->impl().test_detailed_tracking();
}

TEST_F(TwoBoxesTest, model)
{
    auto result = this->summarize_model();
    GenericGeoModelInp ref;
    ref.volume.labels = {"inner", "world"};
    ref.volume.materials = {-1, -1};
    ref.volume.daughters = {{}, {1}};
    ref.volume_instance.labels = {"world_PV", "inner_PV"};
    ref.volume_instance.volumes = {1, 0};
    ref.world = "world";
    EXPECT_REF_EQ(ref, result);
}

TEST_F(TwoBoxesTest, reentrant)
{
    this->impl().test_reentrant();
}

TEST_F(TwoBoxesTest, reentrant_undo)
{
    this->impl().test_reentrant_undo();
}

TEST_F(TwoBoxesTest, tangent)
{
    this->impl().test_tangent();
}

TEST_F(TwoBoxesTest, trace)
{
    this->impl().test_trace();
}

//---------------------------------------------------------------------------//
using ZnenvTest = GenericGeoParameterizedTest<GeantGeoTest, ZnenvGeoTest>;

TEST_F(ZnenvTest, model)
{
    auto result = this->summarize_model();
    GenericGeoModelInp ref;
    ref.volume.labels = {
        "ZNF1",
        "ZNG1",
        "ZNF2",
        "ZNG2",
        "ZNF3",
        "ZNG3",
        "ZNF4",
        "ZNG4",
        "ZNST",
        "ZNSL",
        "ZN1",
        "ZNTX",
        "ZNEU",
        "ZNENV",
        "World",
    };
    ref.volume.materials = {0, 1, 0, 1, 0, 1, 0, 1, 2, 2, 2, 2, 2, 3, 3};
    ref.volume.daughters = {
        {},
        {30},
        {},
        {32},
        {},
        {34},
        {},
        {36},
        {29, 31, 33, 35},
        {18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28},
        {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17},
        {5, 6},
        {3, 4},
        {2},
        {1},
    };
    ref.volume_instance.labels = {
        "World_PV",  "WorldBoxPV", "ZNEU_1",     "ZNTX_PV@0",  "ZNTX_PV@1",
        "ZN1_PV@0",  "ZN1_PV@1",   "ZNSL_PV@0",  "ZNSL_PV@1",  "ZNSL_PV@2",
        "ZNSL_PV@3", "ZNSL_PV@4",  "ZNSL_PV@5",  "ZNSL_PV@6",  "ZNSL_PV@7",
        "ZNSL_PV@8", "ZNSL_PV@9",  "ZNSL_PV@10", "ZNST_PV@0",  "ZNST_PV@1",
        "ZNST_PV@2", "ZNST_PV@3",  "ZNST_PV@4",  "ZNST_PV@5",  "ZNST_PV@6",
        "ZNST_PV@7", "ZNST_PV@8",  "ZNST_PV@9",  "ZNST_PV@10", "ZNG1_1",
        "ZNF1_1",    "ZNG2_1",     "ZNF2_1",     "ZNG3_1",     "ZNF3_1",
        "ZNG4_1",    "ZNF4_1",
    };
    ref.volume_instance.volumes = {
        14, 13, 12, 11, 11, 10, 10, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 8,
        8,  8,  8,  8,  8,  8,  8,  8, 8, 8, 1, 0, 3, 2, 5, 4, 7, 6,
    };
    ref.world = "World";
    EXPECT_REF_EQ(ref, result);
}

TEST_F(ZnenvTest, trace)
{
    this->impl().test_trace();
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
