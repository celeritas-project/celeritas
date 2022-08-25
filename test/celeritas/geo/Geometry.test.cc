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
    return {};
    // return make_span(paths);
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
    return {};
    // return make_span(paths);
}

//---------------------------------------------------------------------------//

#if CELERITAS_USE_VECGEOM
// TODO: ORANGE and VecGeom disagree on path lengths only for this geometry...
#    define ThreeSpheresTest DISABLED_ThreeSpheresTest
#endif

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
    return {}; // make_span(paths);
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
