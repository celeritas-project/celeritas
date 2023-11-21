//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/geo/Geometry.test.cc
//---------------------------------------------------------------------------//
#include "celeritas_config.h"
#include "celeritas/geo/GeoParams.hh"
#include "celeritas/geo/GeoParamsOutput.hh"

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
        return result;
    }

    size_type num_steps() const final { return 1024; }
    SpanConstStr reference_volumes() const final;
    SpanConstReal reference_avg_path() const final;
};

auto TestEm3Test::reference_volumes() const -> SpanConstStr
{
    static const std::string vols[]
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
    static real_type const orange_paths[] = {
        8.617,  0.1079, 0.2445, 0.1325, 0.3059, 0.1198, 0.3179, 0.1258, 0.2803,
        0.1793, 0.4181, 0.1638, 0.3871, 0.1701, 0.3309, 0.1706, 0.4996, 0.217,
        0.5396, 0.188,  0.3784, 0.1758, 0.5439, 0.3073, 0.558,  0.2541, 0.5678,
        0.2246, 0.5444, 0.3164, 0.6882, 0.3186, 0.6383, 0.2305, 0.6078, 0.2813,
        0.6735, 0.278,  0.6635, 0.2961, 0.751,  0.3612, 0.7456, 0.319,  0.7395,
        0.3557, 0.9123, 0.4129, 0.7772, 0.3561, 0.8535, 0.4207, 0.8974, 0.3674,
        0.9314, 0.4164, 0.802,  0.3904, 0.7931, 0.3276, 0.712,  0.3259, 0.7212,
        0.2625, 0.6184, 0.2654, 0.6822, 0.3073, 0.7249, 0.3216, 0.8117, 0.3101,
        0.6056, 0.2372, 0.5184, 0.1985, 0.636,  0.3183, 0.6663, 0.2881, 0.725,
        0.2805, 0.6565, 0.2635, 0.529,  0.2416, 0.5596, 0.2547, 0.5676, 0.2178,
        0.5719, 0.2943, 0.5682, 0.2161, 0.4803, 0.1719, 0.4611, 0.1928, 0.3685,
        0.1554, 0.2443};
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
        return result;
    }

    size_type num_steps() const final { return 1024; }
    SpanConstStr reference_volumes() const final;
    SpanConstReal reference_avg_path() const final;
};

auto SimpleCmsTest::reference_volumes() const -> SpanConstStr
{
    static const std::string vols[] = {"vacuum_tube",
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
    static const real_type paths[]
        = {58.02, 408.3, 274.9, 540.9, 496.3, 1134, 1694};
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
    static const std::string vols[] = {"inner", "middle", "outer", "world"};
    return make_span(vols);
}

auto ThreeSpheresTest::reference_avg_path() const -> SpanConstReal
{
    static const real_type paths[] = {0.2013, 3.346, 6.696, 375.5};
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
    static real_type const paths[] = {74.17136, 13.25306, 76.67924, 449.5464,
        0.09551618, 0.3231404, 0.310899, 0.3844357, 0.01179415, 11.09485,
        9.101073, 0.0004083249, 0.3033329, 0.4292332, 228.7892, 0.03947559,
        563.0746, 2858.592};
    // clang-format on
    return make_span(paths);
}

//---------------------------------------------------------------------------//
// TESTEM3
//---------------------------------------------------------------------------//

TEST_F(TestEm3Test, host)
{
    if (not_orange_geo)
    {
        EXPECT_TRUE(this->geometry()->supports_safety());
    }
    else
    {
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

    if (!CELERITAS_USE_JSON)
    {
        EXPECT_EQ(R"json("output unavailable")json", to_string(out));
    }
    else if (CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_VECGEOM)
    {
        EXPECT_EQ(
            R"json({"bbox":[[-1000.001,-1000.001,-2000.001],[1000.001,1000.001,2000.001]],"supports_safety":true,"volumes":{"label":["vacuum_tube","si_tracker","em_calorimeter","had_calorimeter","sc_solenoid","fe_muon_chambers","world"]}})json",
            to_string(out))
            << "\n/*** REPLACE ***/\nR\"json(" << to_string(out)
            << ")json\"\n/******/";
    }
    else if (CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_ORANGE)
    {
        EXPECT_EQ(
            R"json({"bbox":[[-1000.0,-1000.0,-2000.0],[1000.0,1000.0,2000.0]],"supports_safety":false,"surfaces":{"label":["world_box.mx@global","world_box.px@global","world_box.my@global","world_box.py@global","world_box.mz@global","world_box.pz@global","guide_tube.coz@global","crystal_em_calorimeter_outer.mz@global","crystal_em_calorimeter_outer.pz@global","silicon_tracker_outer.coz@global","crystal_em_calorimeter_outer.coz@global","hadron_calorimeter_outer.coz@global","superconducting_solenoid_outer.coz@global","iron_muon_chambers_outer.coz@global"]},"volumes":{"label":["[EXTERIOR]@global","vacuum_tube@global","si_tracker@global","em_calorimeter@global","had_calorimeter@global","sc_solenoid@global","fe_muon_chambers@global","world@global"]}})json",
            to_string(out))
            << "\n/*** REPLACE ***/\nR\"json(" << to_string(out)
            << ")json\"\n/******/";
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

TEST_F(CmseTest, host)
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
