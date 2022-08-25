//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/geo/Geometry.test.cc
//---------------------------------------------------------------------------//
#include "celeritas_config.h"
#include "celeritas/geo/GeoParams.hh"

#include "HeuristicGeoTestBase.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class TestEm3Test : public HeuristicGeoTestBase
{
  protected:
    const char* geometry_basename() const override { return "testem3-flat"; }

    HeuristicGeoScalars build_scalars() const final
    {
        HeuristicGeoScalars result;
        result.lower = {-19.77, -20, -20};
        result.upper = {19.43, 20, 20};
        return result;
    }

    size_type     num_steps() const final { return 1024; }
    SpanConstStr  reference_volumes() const final;
    SpanConstReal reference_avg_path() const final;
};

auto TestEm3Test::reference_volumes() const -> SpanConstStr
{
    static const std::string vols[]
        = {"world_lv",       "gap_lv_0",  "absorber_lv_0",  "gap_lv_1",
           "absorber_lv_1",  "gap_lv_2",  "absorber_lv_2",  "gap_lv_3",
           "absorber_lv_3",  "gap_lv_4",  "absorber_lv_4",  "gap_lv_5",
           "absorber_lv_5",  "gap_lv_6",  "absorber_lv_6",  "gap_lv_7",
           "absorber_lv_7",  "gap_lv_8",  "absorber_lv_8",  "gap_lv_9",
           "absorber_lv_9",  "gap_lv_10", "absorber_lv_10", "gap_lv_11",
           "absorber_lv_11", "gap_lv_12", "absorber_lv_12", "gap_lv_13",
           "absorber_lv_13", "gap_lv_14", "absorber_lv_14", "gap_lv_15",
           "absorber_lv_15", "gap_lv_16", "absorber_lv_16", "gap_lv_17",
           "absorber_lv_17", "gap_lv_18", "absorber_lv_18", "gap_lv_19",
           "absorber_lv_19", "gap_lv_20", "absorber_lv_20", "gap_lv_21",
           "absorber_lv_21", "gap_lv_22", "absorber_lv_22", "gap_lv_23",
           "absorber_lv_23", "gap_lv_24", "absorber_lv_24", "gap_lv_25",
           "absorber_lv_25", "gap_lv_26", "absorber_lv_26", "gap_lv_27",
           "absorber_lv_27", "gap_lv_28", "absorber_lv_28", "gap_lv_29",
           "absorber_lv_29", "gap_lv_30", "absorber_lv_30", "gap_lv_31",
           "absorber_lv_31", "gap_lv_32", "absorber_lv_32", "gap_lv_33",
           "absorber_lv_33", "gap_lv_34", "absorber_lv_34", "gap_lv_35",
           "absorber_lv_35", "gap_lv_36", "absorber_lv_36", "gap_lv_37",
           "absorber_lv_37", "gap_lv_38", "absorber_lv_38", "gap_lv_39",
           "absorber_lv_39", "gap_lv_40", "absorber_lv_40", "gap_lv_41",
           "absorber_lv_41", "gap_lv_42", "absorber_lv_42", "gap_lv_43",
           "absorber_lv_43", "gap_lv_44", "absorber_lv_44", "gap_lv_45",
           "absorber_lv_45", "gap_lv_46", "absorber_lv_46", "gap_lv_47",
           "absorber_lv_47", "gap_lv_48", "absorber_lv_48", "gap_lv_49",
           "absorber_lv_49"};
    return make_span(vols);
}

auto TestEm3Test::reference_avg_path() const -> SpanConstReal
{
    static const real_type paths[] = {
        8.265,  0.1061, 0.2384, 0.1297, 0.2994, 0.124,  0.3086, 0.1157, 0.2718,
        0.1735, 0.3806, 0.1685, 0.3884, 0.1712, 0.398,  0.1677, 0.499,  0.2431,
        0.6357, 0.2107, 0.4141, 0.1788, 0.5612, 0.3101, 0.5513, 0.2539, 0.579,
        0.2249, 0.531,  0.2885, 0.6549, 0.3089, 0.7117, 0.2505, 0.6564, 0.3055,
        0.647,  0.2944, 0.6914, 0.3151, 0.8069, 0.3401, 0.802,  0.3504, 0.771,
        0.376,  0.9649, 0.3929, 0.8178, 0.3676, 0.895,  0.424,  0.9008, 0.3821,
        0.8925, 0.4202, 0.764,  0.3841, 0.7901, 0.3464, 0.747,  0.2901, 0.6771,
        0.2334, 0.5833, 0.2591, 0.659,  0.2893, 0.6955, 0.317,  0.7173, 0.3035,
        0.6043, 0.2588, 0.5531, 0.2089, 0.6115, 0.3052, 0.6891, 0.3069, 0.7394,
        0.2706, 0.6481, 0.2451, 0.504,  0.1975, 0.5367, 0.2372, 0.557,  0.2162,
        0.5028, 0.2771, 0.4707, 0.2246, 0.4944, 0.1799, 0.5255, 0.2078, 0.3951,
        0.171,  0.274};
    return make_span(paths);
}

//---------------------------------------------------------------------------//

class SimpleCmsTest : public HeuristicGeoTestBase
{
  protected:
    const char* geometry_basename() const override { return "simple-cms"; }

    HeuristicGeoScalars build_scalars() const final
    {
        HeuristicGeoScalars result;
        result.lower = {-30, -30, -700};
        result.upper = {30, 30, 700};
        result.log_min_step = std::log(1e-4);
        result.log_max_step = std::log(1e2);
        return result;
    }

    size_type     num_steps() const final { return 1024; }
    SpanConstStr  reference_volumes() const final;
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
        = {57.5, 406, 271, 534, 486, 1.16e+03, 1.7e+03};
    return make_span(paths);
}

//---------------------------------------------------------------------------//

class ThreeSpheresTest : public HeuristicGeoTestBase
{
  protected:
    const char* geometry_basename() const override { return "three-spheres"; }

    HeuristicGeoScalars build_scalars() const final
    {
        HeuristicGeoScalars result;
        result.lower = {-2.1, -2.1, -2.1};
        result.upper = {2.1, 2.1, 2.1};
        return result;
    }

    size_type     num_steps() const final { return 1024; }
    SpanConstStr  reference_volumes() const final;
    SpanConstReal reference_avg_path() const final;
};

auto ThreeSpheresTest::reference_volumes() const -> SpanConstStr
{
    static const std::string vols[] = {"inner", "middle", "outer", "world"};
    return make_span(vols);
}

auto ThreeSpheresTest::reference_avg_path() const -> SpanConstReal
{
    static const real_type paths[] = {126, 342, 22.6, 3.69, 0, 0, 7.88};
    return make_span(paths);
}

//---------------------------------------------------------------------------//
// TESTEM3
//---------------------------------------------------------------------------//

TEST_F(TestEm3Test, host)
{
    if (CELERITAS_USE_VECGEOM)
    {
        EXPECT_TRUE(this->geometry()->supports_safety());
    }
    else
    {
        EXPECT_FALSE(this->geometry()->supports_safety());
    }
    this->run_host(512, 1e-3);
}

TEST_F(TestEm3Test, TEST_IF_CELER_DEVICE(device))
{
    this->run_device(512, 1e-3);
}

//---------------------------------------------------------------------------//
// SIMPLECMS
//---------------------------------------------------------------------------//

TEST_F(SimpleCmsTest, host)
{
    real_type tol = (CELERITAS_USE_VECGEOM ? 2e-2 : 1e-3);
    this->run_host(512, tol);
}

TEST_F(SimpleCmsTest, TEST_IF_CELER_DEVICE(device))
{
    real_type tol = (CELERITAS_USE_VECGEOM ? 2e-2 : 1e-3);
    this->run_device(512, tol);
}

//---------------------------------------------------------------------------//
// THREE_SPHERES
//---------------------------------------------------------------------------//

TEST_F(ThreeSpheresTest, host)
{
    EXPECT_TRUE(this->geometry()->supports_safety());
    this->run_host(512, 1e-3);
}

TEST_F(ThreeSpheresTest, TEST_IF_CELER_DEVICE(device))
{
    this->run_device(512, 1e-3);
}

//---------------------------------------------------------------------------//
} // namespace test
} // namespace celeritas
