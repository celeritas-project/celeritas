//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/geo/Geometry.test.cc
//---------------------------------------------------------------------------//
#include "corecel/Config.hh"

#include "geocel/GeoParamsOutput.hh"
#include "celeritas/geo/GeoParams.hh"

#include "HeuristicGeoTestBase.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
constexpr bool not_orange_geo
    = (CELERITAS_CORE_GEO != CELERITAS_CORE_GEO_ORANGE);
//---------------------------------------------------------------------------//

class TestEm3Test : public HeuristicGeoTestBase
{
  protected:
    std::string_view geometry_basename() const override
    {
        return "testem3-flat"sv;
    }

    HeuristicGeoScalars build_scalars() const final
    {
        HeuristicGeoScalars result;
        result.lower = {-19.77, -20, -20};
        result.upper = {19.43, 20, 20};
        result.world_volume = this->geometry()->volumes().find_unique("world");
        return result;
    }

    size_type num_steps() const final { return 1024; }
    SpanConstStr reference_volumes() const final;
    SpanConstReal reference_avg_path() const final;
};

auto TestEm3Test::reference_volumes() const -> SpanConstStr
{
    static std::string const vols[]
        = {"world",       "gap_0",  "absorber_0",  "gap_1",
           "absorber_1",  "gap_2",  "absorber_2",  "gap_3",
           "absorber_3",  "gap_4",  "absorber_4",  "gap_5",
           "absorber_5",  "gap_6",  "absorber_6",  "gap_7",
           "absorber_7",  "gap_8",  "absorber_8",  "gap_9",
           "absorber_9",  "gap_10", "absorber_10", "gap_11",
           "absorber_11", "gap_12", "absorber_12", "gap_13",
           "absorber_13", "gap_14", "absorber_14", "gap_15",
           "absorber_15", "gap_16", "absorber_16", "gap_17",
           "absorber_17", "gap_18", "absorber_18", "gap_19",
           "absorber_19", "gap_20", "absorber_20", "gap_21",
           "absorber_21", "gap_22", "absorber_22", "gap_23",
           "absorber_23", "gap_24", "absorber_24", "gap_25",
           "absorber_25", "gap_26", "absorber_26", "gap_27",
           "absorber_27", "gap_28", "absorber_28", "gap_29",
           "absorber_29", "gap_30", "absorber_30", "gap_31",
           "absorber_31", "gap_32", "absorber_32", "gap_33",
           "absorber_33", "gap_34", "absorber_34", "gap_35",
           "absorber_35", "gap_36", "absorber_36", "gap_37",
           "absorber_37", "gap_38", "absorber_38", "gap_39",
           "absorber_39", "gap_40", "absorber_40", "gap_41",
           "absorber_41", "gap_42", "absorber_42", "gap_43",
           "absorber_43", "gap_44", "absorber_44", "gap_45",
           "absorber_45", "gap_46", "absorber_46", "gap_47",
           "absorber_47", "gap_48", "absorber_48", "gap_49",
           "absorber_49"};
    return make_span(vols);
}

auto TestEm3Test::reference_avg_path() const -> SpanConstReal
{
    static real_type const orange_paths[]
        = {7.504,  0.07378, 0.2057, 0.102,  0.2408, 0.1006, 0.3019, 0.1153,
           0.2812, 0.1774,  0.4032, 0.1354, 0.3163, 0.1673, 0.3465, 0.1786,
           0.4494, 0.2237,  0.5863, 0.192,  0.4027, 0.1905, 0.5949, 0.3056,
           0.5217, 0.2179,  0.5365, 0.2123, 0.5484, 0.2938, 0.634,  0.3144,
           0.6364, 0.2207,  0.5688, 0.2685, 0.6717, 0.2697, 0.6468, 0.2824,
           0.7424, 0.3395,  0.6919, 0.3018, 0.7078, 0.3441, 0.9093, 0.4125,
           0.7614, 0.334,   0.8102, 0.3901, 0.8114, 0.3377, 0.8856, 0.39,
           0.7765, 0.3847,  0.785,  0.3017, 0.6694, 0.3026, 0.7018, 0.2482,
           0.6192, 0.2405,  0.6014, 0.2733, 0.6454, 0.2804, 0.6941, 0.2608,
           0.5855, 0.2318,  0.5043, 0.1906, 0.6139, 0.3125, 0.6684, 0.3002,
           0.7295, 0.2874,  0.6328, 0.2524, 0.532,  0.2354, 0.5435, 0.2612,
           0.5484, 0.2294,  0.5671, 0.246,  0.5157, 0.1953, 0.3996, 0.1301,
           0.3726, 0.1642,  0.3317, 0.1375, 0.1909};
    return make_span(orange_paths);
}

//---------------------------------------------------------------------------//

class SimpleCmsTest : public HeuristicGeoTestBase
{
  protected:
    std::string_view geometry_basename() const override
    {
        return "simple-cms"sv;
    }

    HeuristicGeoScalars build_scalars() const final
    {
        HeuristicGeoScalars result;
        result.lower = {-30, -30, -700};
        result.upper = {30, 30, 700};
        result.log_min_step = std::log(1e-4);
        result.log_max_step = std::log(1e2);
        result.world_volume = this->geometry()->volumes().find_unique("world");
        return result;
    }

    size_type num_steps() const final { return 1024; }
    SpanConstStr reference_volumes() const final;
    SpanConstReal reference_avg_path() const final;
};

auto SimpleCmsTest::reference_volumes() const -> SpanConstStr
{
    static std::string const vols[] = {"vacuum_tube",
                                       "si_tracker",
                                       "em_calorimeter",
                                       "had_calorimeter",
                                       "sc_solenoid",
                                       "fe_muon_chambers",
                                       "world"};
    return make_span(vols);
}

auto SimpleCmsTest::reference_avg_path() const -> SpanConstReal
{
    static real_type const paths[]
        = {56.38, 403, 261.3, 507.5, 467.1, 1142, 1851};
    return make_span(paths);
}

//---------------------------------------------------------------------------//

class ThreeSpheresTest : public HeuristicGeoTestBase
{
  protected:
    std::string_view geometry_basename() const override
    {
        return "three-spheres"sv;
    }

    HeuristicGeoScalars build_scalars() const final
    {
        HeuristicGeoScalars result;
        result.lower = {-2.1, -2.1, -2.1};
        result.upper = {2.1, 2.1, 2.1};
        return result;
    }

    size_type num_steps() const final { return 1024; }
    SpanConstStr reference_volumes() const final;
    SpanConstReal reference_avg_path() const final;
};

auto ThreeSpheresTest::reference_volumes() const -> SpanConstStr
{
    static std::string const vols[] = {"inner", "middle", "outer", "world"};
    return make_span(vols);
}

auto ThreeSpheresTest::reference_avg_path() const -> SpanConstReal
{
    static real_type const paths[] = {0.2013, 3.346, 6.696, 375.5};
    return make_span(paths);
}

//---------------------------------------------------------------------------//

#if CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_ORANGE
#    define CmseTest DISABLED_CmseTest
#endif
class CmseTest : public HeuristicGeoTestBase
{
  protected:
    std::string_view geometry_basename() const override { return "cmse"sv; }

    HeuristicGeoScalars build_scalars() const final
    {
        HeuristicGeoScalars result;
        result.lower = {-80, -80, -4500};
        result.upper = {80, 80, 4500};
        result.log_min_step = std::log(1e-4);
        result.log_max_step = std::log(1e3);
        return result;
    }

    size_type num_steps() const final { return 1024; }
    SpanConstStr reference_volumes() const final;
    SpanConstReal reference_avg_path() const final;
};

auto CmseTest::reference_volumes() const -> SpanConstStr
{
    // clang-format off
    static std::string const vols[] = {"CMStoZDC", "Tracker", "CALO", "MUON",
        "BEAM", "BEAM1", "BEAM2", "BEAM3", "TrackerPixelNose", "VCAL",
        "TotemT1", "TotemT2", "CastorF", "CastorB", "OQUA", "BSC2", "CMSE",
        "OCMS"};
    // clang-format on
    return make_span(vols);
}

auto CmseTest::reference_avg_path() const -> SpanConstReal
{
    // clang-format off
    static real_type const paths[] = {
       74.681789113, 13.9060168654525, 67.789037081, 460.34598500,
       0.0752032527, 0.3958262271, 0.25837963337, 0.51484801201, 0.01179415,
       10.662958365, 9.3044714865, 0.0004083249, 0.25874352886, 0.4292332,
       225.390314534812, 0.0394755943, 550.75653646, 2824.1066316 };
    // clang-format on
    return make_span(paths);
}

//---------------------------------------------------------------------------//
// TESTEM3
//---------------------------------------------------------------------------//

TEST_F(TestEm3Test, host)
{
    if (CELERITAS_USE_GEANT4 || CELERITAS_CORE_GEO != CELERITAS_CORE_GEO_ORANGE)
    {
        EXPECT_TRUE(this->geometry()->supports_safety());
    }
    else
    {
        // ORANGE from JSON file doesn't support safety
        EXPECT_FALSE(this->geometry()->supports_safety());
    }
    real_type tol = not_orange_geo ? 0.35 : 1e-3;
    this->run_host(512, tol);
}

TEST_F(TestEm3Test, TEST_IF_CELER_DEVICE(device))
{
    real_type tol = not_orange_geo ? 0.25 : 1e-3;
    this->run_device(512, tol);
}

//---------------------------------------------------------------------------//
// SIMPLECMS
//---------------------------------------------------------------------------//

TEST_F(SimpleCmsTest, host)
{
    // Results were generated with ORANGE
    real_type tol = not_orange_geo ? 0.05 : 1e-3;
    this->run_host(512, tol);
}

TEST_F(SimpleCmsTest, TEST_IF_CELER_DEVICE(device))
{
    // Results were generated with ORANGE
    real_type tol = not_orange_geo ? 0.025 : 1e-3;
    this->run_device(512, tol);
}

TEST_F(SimpleCmsTest, output)
{
    GeoParamsOutput out(this->geometry());
    EXPECT_EQ("geometry", out.label());

    if (CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_VECGEOM)
    {
        EXPECT_JSON_EQ(
            R"json({"_category":"internal","_label":"geometry","bbox":[[-1000.001,-1000.001,-2000.001],[1000.001,1000.001,2000.001]],"max_depth":2,"supports_safety":true,"volumes":{"label":["vacuum_tube","si_tracker","em_calorimeter","had_calorimeter","sc_solenoid","fe_muon_chambers","world"]}})json",
            to_string(out));
    }
    else if (CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_ORANGE
             && CELERITAS_USE_GEANT4)
    {
        EXPECT_JSON_EQ(
            R"json({"_category":"internal","_label":"geometry","bbox":[[-1000.0,-1000.0,-2000.0],[1000.0,1000.0,2000.0]],"max_depth":1,"supports_safety":false,"surfaces":{"label":["world_box@mx","world_box@px","world_box@my","world_box@py","world_box@mz","world_box@pz","crystal_em_calorimeter@excluded.mz","crystal_em_calorimeter@excluded.pz","lhc_vacuum_tube@cz","crystal_em_calorimeter@excluded.cz","crystal_em_calorimeter@interior.cz","hadron_calorimeter@interior.cz","iron_muon_chambers@excluded.cz","iron_muon_chambers@interior.cz"]},"volumes":{"label":["[EXTERIOR]@world","vacuum_tube@world","si_tracker@world","em_calorimeter@world","had_calorimeter@world","sc_solenoid@world","fe_muon_chambers@world","world@world"]}})json",
            this->genericize_pointers(to_string(out)));
    }
    else if (CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_ORANGE)
    {
        EXPECT_JSON_EQ(
            R"json({"_category":"internal","_label":"geometry","bbox":[[-1000.0,-1000.0,-2000.0],[1000.0,1000.0,2000.0]],"max_depth":1,"supports_safety":false,"surfaces":{"label":["world_box.mx@global","world_box.px@global","world_box.my@global","world_box.py@global","world_box.mz@global","world_box.pz@global","guide_tube.coz@global","crystal_em_calorimeter_outer.mz@global","crystal_em_calorimeter_outer.pz@global","silicon_tracker_outer.coz@global","crystal_em_calorimeter_outer.coz@global","hadron_calorimeter_outer.coz@global","superconducting_solenoid_outer.coz@global","iron_muon_chambers_outer.coz@global"]},"volumes":{"label":["[EXTERIOR]@global","vacuum_tube@global","si_tracker@global","em_calorimeter@global","had_calorimeter@global","sc_solenoid@global","fe_muon_chambers@global","world@global"]}})json",
            this->genericize_pointers(to_string(out)));
    }
}

//---------------------------------------------------------------------------//
// THREE_SPHERES
//---------------------------------------------------------------------------//

TEST_F(ThreeSpheresTest, host)
{
    // Results were generated with ORANGE
    real_type tol = not_orange_geo ? 0.05 : 1e-3;
    EXPECT_TRUE(this->geometry()->supports_safety());
    this->run_host(512, tol);
}

TEST_F(ThreeSpheresTest, TEST_IF_CELER_DEVICE(device))
{
    // Results were generated with ORANGE
    real_type tol = not_orange_geo ? 0.025 : 1e-3;
    this->run_device(512, tol);
}

//---------------------------------------------------------------------------//
// CMSE
//---------------------------------------------------------------------------//
// TODO: ensure reference values are the same for all CI platforms (see #1570)
TEST_F(CmseTest, DISABLED_host)
{
    auto const& bbox = this->geometry()->bbox();
    real_type const geo_eps
        = CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_VECGEOM ? 0.001 : 0;
    EXPECT_VEC_SOFT_EQ(
        (Real3{-1750 - geo_eps, -1750 - geo_eps, -45000 - geo_eps}),
        bbox.lower());
    EXPECT_VEC_SOFT_EQ((Real3{1750 + geo_eps, 1750 + geo_eps, 45000 + geo_eps}),
                       bbox.upper());

    real_type tol = CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_VECGEOM ? 0.005
                                                                     : 0.35;
    this->run_host(512, tol);
}

TEST_F(CmseTest, TEST_IF_CELER_DEVICE(device))
{
    this->run_device(512, 0.005);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
