//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/ReplicaGeoTest.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string_view>

#include "corecel/math/Turn.hh"

#include "GenericGeoTestInterface.hh"
#include "TestMacros.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Test the B5 (replica) geometry.
 */
class ReplicaGeoTest
{
  public:
    static std::string_view geometry_basename() { return "replica"; }

    //! Construct with a reference to the GoogleTest
    ReplicaGeoTest(GenericGeoTestInterface* geo_test) : test_{geo_test} {}

    inline void test_trace() const;
    inline void test_volume_stack() const;

  private:
    GenericGeoTestInterface* test_;
};

//---------------------------------------------------------------------------//
void ReplicaGeoTest::test_trace() const
{
    {
        SCOPED_TRACE("Center +z");
        auto result = test_->track({0, 0.1, -990}, {0, 0, 1});
        real_type const safety_tol = test_->safety_tol();
        static char const* const expected_volumes[] = {
            "world",    "firstArm",   "hodoscope1", "firstArm",
            "chamber1", "wirePlane1", "chamber1",   "firstArm",
            "chamber1", "wirePlane1", "chamber1",   "firstArm",
            "chamber1", "wirePlane1", "chamber1",   "firstArm",
            "chamber1", "wirePlane1", "chamber1",   "firstArm",
            "chamber1", "wirePlane1", "chamber1",   "firstArm",
            "world",    "magnetic",   "world",      "secondArm",
            "chamber2", "wirePlane2", "chamber2",   "secondArm",
            "world",
        };
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static char const* const expected_volume_instances[] = {
            "world_PV", "firstArm",   "hodoscope1", "firstArm",
            "chamber1", "wirePlane1", "chamber1",   "firstArm",
            "chamber1", "wirePlane1", "chamber1",   "firstArm",
            "chamber1", "wirePlane1", "chamber1",   "firstArm",
            "chamber1", "wirePlane1", "chamber1",   "firstArm",
            "chamber1", "wirePlane1", "chamber1",   "firstArm",
            "world_PV", "magnetic",   "world_PV",   "fSecondArmPhys",
            "chamber2", "wirePlane2", "chamber2",   "fSecondArmPhys",
            "world_PV",
        };
        EXPECT_VEC_EQ(expected_volume_instances, result.volume_instances);
        static real_type const expected_distances[] = {
            190,
            149.5,
            1,
            48.5,
            0.99,
            0.020000000000036,
            0.98999999999996,
            48,
            0.99,
            0.020000000000036,
            0.98999999999996,
            48,
            0.99,
            0.020000000000036,
            0.98999999999996,
            48,
            0.99,
            0.020000000000036,
            0.98999999999996,
            48,
            0.99,
            0.019999999999991,
            0.99000000000001,
            199,
            100,
            200,
            73.205080756887,
            114.31535329955,
            1.1431535329955,
            0.023094010767522,
            1.1431535329955,
            110.17016486681,
            600,
        };
        EXPECT_VEC_NEAR(expected_distances, result.distances, 1e-11);
        static real_type const expected_hw_safety[] = {
            95,
            74.75,
            0.5,
            24.25,
            0.49499999999998,
            0.01,
            0.49499999999998,
            24,
            0.49499999999998,
            0.01,
            0.49499999999998,
            24,
            0.49499999999998,
            0.01,
            0.49499999999998,
            24,
            0.49499999999998,
            0.01,
            0.49499999999998,
            24,
            0.49499999999998,
            0.01,
            0.49499999999998,
            99.5,
            50,
            99.9,
            31.698729810778,
            49.5,
            0.49499999999998,
            0.01,
            0.49499999999997,
            22.457458783298,
            150,
        };
        EXPECT_VEC_NEAR(
            expected_hw_safety, result.halfway_safeties, safety_tol);
    }
    {
        SCOPED_TRACE("Second arm");
        Real3 dir{0, 0, 0};
        sincos(Turn{-30.0 / 360.}, &dir[0], &dir[2]);
        auto result = test_->track({0, 0.1, 0}, dir);
        real_type const safety_tol = test_->safety_tol();
        static char const* const expected_volumes[] = {
            "magnetic",    "world",       "secondArm",   "chamber2",
            "wirePlane2",  "chamber2",    "secondArm",   "chamber2",
            "wirePlane2",  "chamber2",    "secondArm",   "chamber2",
            "wirePlane2",  "chamber2",    "secondArm",   "chamber2",
            "wirePlane2",  "chamber2",    "secondArm",   "chamber2",
            "wirePlane2",  "chamber2",    "secondArm",   "hodoscope2",
            "secondArm",   "cell",        "secondArm",   "HadCalLayer",
            "HadCalLayer", "HadCalLayer", "HadCalLayer", "HadCalLayer",
            "HadCalLayer", "HadCalLayer", "HadCalLayer", "HadCalLayer",
            "HadCalLayer", "HadCalLayer", "HadCalLayer", "HadCalLayer",
            "HadCalLayer", "HadCalLayer", "HadCalLayer", "HadCalLayer",
            "HadCalLayer", "HadCalLayer", "HadCalLayer", "world",
        };
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static char const* const expected_volume_instances[] = {
            "magnetic",          "world_PV",          "fSecondArmPhys",
            "chamber2",          "wirePlane2",        "chamber2",
            "fSecondArmPhys",    "chamber2",          "wirePlane2",
            "chamber2",          "fSecondArmPhys",    "chamber2",
            "wirePlane2",        "chamber2",          "fSecondArmPhys",
            "chamber2",          "wirePlane2",        "chamber2",
            "fSecondArmPhys",    "chamber2",          "wirePlane2",
            "chamber2",          "fSecondArmPhys",    "hodoscope2",
            "fSecondArmPhys",    "cell_param@42",     "fSecondArmPhys",
            "HadCalLayer_PV@0",  "HadCalLayer_PV@1",  "HadCalLayer_PV@2",
            "HadCalLayer_PV@3",  "HadCalLayer_PV@4",  "HadCalLayer_PV@5",
            "HadCalLayer_PV@6",  "HadCalLayer_PV@7",  "HadCalLayer_PV@8",
            "HadCalLayer_PV@9",  "HadCalLayer_PV@10", "HadCalLayer_PV@11",
            "HadCalLayer_PV@12", "HadCalLayer_PV@13", "HadCalLayer_PV@14",
            "HadCalLayer_PV@15", "HadCalLayer_PV@16", "HadCalLayer_PV@17",
            "HadCalLayer_PV@18", "HadCalLayer_PV@19", "world_PV",
        };
        EXPECT_VEC_EQ(expected_volume_instances, result.volume_instances);
        static real_type const expected_distances[] = {
            100,
            50,
            99,
            0.99000000000001,
            0.019999999999995,
            0.99000000000002,
            48,
            0.99,
            0.01999999999998,
            0.99,
            48,
            0.99,
            0.01999999999998,
            0.99,
            48,
            0.99,
            0.01999999999998,
            0.99,
            48,
            0.99,
            0.01999999999998,
            0.99,
            48.5,
            0.99999999999999,
            184.5,
            30,
            35,
            5,
            4.9999999999999,
            5.0000000000001,
            5,
            5,
            5,
            5,
            4.9999999999999,
            5.0000000000001,
            5,
            5,
            5,
            5,
            4.9999999999999,
            5.0000000000001,
            5,
            5,
            5,
            5,
            4.9999999999999,
            304.70053837925,
        };
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
        static real_type const expected_hw_safety[] = {
            50,    25,
            49.5,  0.49499999999998,
            0.01,  0.49499999999997,
            24,    0.49499999999998,
            0.01,  0.49499999999997,
            24,    0.49499999999998,
            0.01,  0.49499999999997,
            24,    0.49499999999998,
            0.01,  0.49499999999997,
            24,    0.49499999999998,
            0.01,  0.49499999999993,
            24.25, 0.5,
            92.25, 0,
            17.5,  0,
            0,     0,
            0,     0,
            0,     0,
            0,     0,
            0,     0,
            0,     0,
            0,     0,
            0,     0,
            0,     0,
            0,     131.93920339161,
        };
        EXPECT_VEC_NEAR(
            expected_hw_safety, result.halfway_safeties, safety_tol);
        if (true)  // if (gest_->geometry_type() == "geant4")
        {
            EXPECT_EQ(0, result.bumps.size()) << repr(result.bumps);
        }
    }
}

//---------------------------------------------------------------------------//
void ReplicaGeoTest::test_volume_stack() const
{
    {
        auto result = test_->volume_stack({-400, 0.1, 650});
        GenericGeoVolumeStackResult ref;
        ref.volume_instances = {
            "world_PV",
            "fSecondArmPhys",
            "HadCalorimeter",
            "HadCalColumn_PV",
            "HadCalCell_PV",
            "HadCalLayer_PV",
        };
        ref.replicas = {-1, -1, -1, 4, 1, 2};
        EXPECT_RESULT_EQ(ref, result);
    }
    {
        auto result = test_->volume_stack({-342.5, 0.0, 593.227402});
        GenericGeoVolumeStackResult ref;
        ref.volume_instances = {
            "world_PV",
            "fSecondArmPhys",
            "EMcalorimeter",
        };
        ref.replicas = {-1, -1, -1};
        EXPECT_RESULT_EQ(ref, result);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
