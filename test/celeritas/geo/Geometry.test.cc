//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/geo/Geometry.test.cc
//---------------------------------------------------------------------------//
#include <gtest/gtest.h>

#include "corecel/Config.hh"

#include "corecel/StringSimplifier.hh"
#include "geocel/GeoParamsOutput.hh"
#include "celeritas/geo/CoreGeoParams.hh"

#include "HeuristicGeoTestBase.hh"
#include "TestMacros.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
constexpr bool using_orange_geo
    = (CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_ORANGE);
constexpr bool using_vg = CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_VECGEOM;

//---------------------------------------------------------------------------//
class GeometryTest : public HeuristicGeoTestBase
{
  public:
    using VecReal = std::vector<real_type>;
    static void TearDownTestSuite() { last_path() = {}; }

    static void compare_against_previous(SpanConstReal values)
    {
        auto& lp = GeometryTest::last_path();
        if (lp.empty())
        {
            lp.assign(values.begin(), values.end());
        }
        else
        {
            EXPECT_VEC_SOFT_EQ(lp, values);
        }
    }

  private:
    static VecReal& last_path()
    {
        static VecReal p;
        return p;
    }
};

class TestEm3Test : public HeuristicGeoTestBase
{
  protected:
    std::string_view gdml_basename() const override
    {
        return "testem3-flat"sv;
    }

    HeuristicGeoScalars build_scalars() const final
    {
        HeuristicGeoScalars result;
        result.lower = {-19.77, -20, -20};
        result.upper = {19.43, 20, 20};
        result.world_volume
            = this->geometry()->impl_volumes().find_unique("world");
        return result;
    }

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
    static real_type const paths[] = {
        7.7662025645238,  0.080436919091118, 0.20906819128204,
        0.10341971435766, 0.24442742004502,  0.10505779938552,
        0.26853647729147, 0.11478190785371,  0.25108494748183,
        0.16275614647623, 0.35684411181979,  0.12070929565832,
        0.2888565798791,  0.17840855709179,  0.34929831766689,
        0.16411790601534, 0.43972046590778,  0.21379238581294,
        0.48619139484194, 0.17494602518841,  0.36077540983427,
        0.18745409879988, 0.51414709415072,  0.2646563441426,
        0.45705298436828, 0.18030129445946,  0.52327385767217,
        0.20544279231036, 0.50553437440921,  0.28801373385678,
        0.65048870681844, 0.31357726664389,  0.59290833575642,
        0.20915394767169, 0.5912207428656,   0.28240450457467,
        0.68907624205949, 0.26910420268894,  0.6606770701755,
        0.30417339698039, 0.75934203294531,  0.32065849027611,
        0.69793532567828, 0.3032329211676,   0.7214124965831,
        0.3389956448183,  0.90203142546933,  0.41314147285831,
        0.78324303836495, 0.36060752044244,  0.79170668197051,
        0.37995675278007, 0.79451581232248,  0.32838821473506,
        0.85330539697848, 0.37601332703705,  0.78408718740086,
        0.38607042473588, 0.78321280196578,  0.31453899989453,
        0.64955173540825, 0.29397539778279,  0.69796517974068,
        0.24525768322878, 0.65006623602054,  0.26616334641623,
        0.67437909504339, 0.28888249357192,  0.66152800260054,
        0.30076215517064, 0.70351910161418,  0.27531434585801,
        0.57324258052408, 0.23250545039233,  0.51275424345496,
        0.2168406554649,  0.60811652275962,  0.31506819796893,
        0.69690066834181, 0.30151441113098,  0.73074573212379,
        0.30074874138257, 0.71067813488393,  0.30726391224123,
        0.59292351490755, 0.23619185409827,  0.57739610782314,
        0.26927647941776, 0.56045055887279,  0.24979059910026,
        0.55821379478737, 0.24840237717025,  0.52234151059082,
        0.18310556267665, 0.3719862592643,   0.12440516234962,
        0.34905658478792, 0.16284436650089,  0.2958888858561,
        0.11815507344671, 0.19055547284288,
    };
    return make_span(paths);
    //(void)paths;
    // return {};
}

//---------------------------------------------------------------------------//

class SimpleCmsTest : public HeuristicGeoTestBase
{
  protected:
    std::string_view gdml_basename() const override { return "simple-cms"sv; }

    HeuristicGeoScalars build_scalars() const final
    {
        HeuristicGeoScalars result;
        result.lower = {-30, -30, -700};
        result.upper = {30, 30, 700};
        result.log_min_step = std::log(1e-4);
        result.log_max_step = std::log(1e2);
        result.world_volume
            = this->geometry()->impl_volumes().find_unique("world");
        return result;
    }

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
    if (CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_VECGEOM)
    {
        static real_type paths[]
            = {56, 390, 255.5, 497.960489118954, 451, 1137, 1870};
        if (using_vg && CELERITAS_VECGEOM_VERSION >= 0x020000)
        {
            // TODO: try to fix any discrepancies from vg2.x
            paths[4] = 454.195842538179;
            paths[5] = 1148.46821493334;
            paths[6] = 1871.43143781146;
        }
        return make_span(paths);
    }
    else
    {
        static real_type const paths[] = {55.1981791404751,
                                          390.86666493102,
                                          255.58787347462,
                                          498.55418963745,
                                          467.44319448399,
                                          1146.725884775,
                                          1865.583699532};
        return make_span(paths);
    }
}

//---------------------------------------------------------------------------//

class ThreeSpheresTest : public HeuristicGeoTestBase
{
  protected:
    std::string_view gdml_basename() const override
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
    static real_type paths[] = {
        0.195837257764839,
        3.28275955815444,
        6.54698622785098,
        376.100451629357,
    };
    if (using_vg && CELERITAS_VECGEOM_VERSION >= 0x020000)
    {
        // TODO: try to fix any discrepancies from vg2.x
        paths[0] = 0.193968509125204;
        paths[2] = 6.63492117919102;
    }
    return make_span(paths);
}

//---------------------------------------------------------------------------//

#if CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_ORANGE
#    define CmseTest DISABLED_CmseTest
#endif
class CmseTest : public HeuristicGeoTestBase
{
  protected:
    std::string_view gdml_basename() const override { return "cmse"sv; }

    HeuristicGeoScalars build_scalars() const final
    {
        HeuristicGeoScalars result;
        result.lower = {-80, -80, -4500};
        result.upper = {80, 80, 4500};
        result.log_min_step = std::log(1e-4);
        result.log_max_step = std::log(1e3);
        return result;
    }

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

TEST_F(TestEm3Test, run)
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

    // Note: with geom_limit=inf, at step 183, track 358 scatters exactly
    // on the boundary. This results in substantial differences between the
    // geometry implementations which is instructive but not useful necessarily
    // for this test.

    // VecGeom solid and ORANGE diverge fairly quickly: this is in part due to
    // bumps.

    real_type tol = using_orange_geo ? 1e-3 : 0.35;
    this->run(512, /* num_steps = */ 1024, tol);
}

//---------------------------------------------------------------------------//
// SIMPLECMS
//---------------------------------------------------------------------------//

TEST_F(SimpleCmsTest, avg_path)
{
    // Results were generated with ORANGE
    real_type tol = using_orange_geo ? 1e-3 : 0.05;
    this->run(512, /* num_steps = */ 1024, tol);
}

TEST_F(SimpleCmsTest, output)
{
    GeoParamsOutput out(this->geometry());
    EXPECT_EQ("geometry", out.label());

    StringSimplifier simplify_str(1);
    auto s = simplify_str(to_string(out));

    if (CELERITAS_CORE_GEO != CELERITAS_CORE_GEO_ORANGE && CELERITAS_USE_GEANT4)
    {
        EXPECT_JSON_EQ(
            R"json({"_category":"internal","_label":"geometry","bbox":[[-1e3,-1e3,-2e3],[1e3,1e3,2e3]],"num_impl_volumes":7,"supports_safety":true})json",
            s);
    }
}

//---------------------------------------------------------------------------//
// THREE_SPHERES
//---------------------------------------------------------------------------//

TEST_F(ThreeSpheresTest, avg_path)
{
    // Results were generated with ORANGE
    // TODO: investigate differences w.r.t. surface model
    real_type tol = using_orange_geo ? 1e-3 : 0.05;
    EXPECT_TRUE(this->geometry()->supports_safety());
    this->run(512, /* num_steps = */ 1024, tol);
}

//---------------------------------------------------------------------------//
// CMSE
//---------------------------------------------------------------------------//
// TODO: ensure reference values are the same for all CI platforms (see #1570)
TEST_F(CmseTest, DISABLED_avg_path)
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
    this->run(512, /* num_steps = */ 1024, tol);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
