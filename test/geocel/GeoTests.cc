//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/GeoTests.cc
//---------------------------------------------------------------------------//
#include "GeoTests.hh"

#include <string_view>

#include "corecel/cont/Range.hh"
#include "corecel/math/Turn.hh"

#include "GenericGeoTestInterface.hh"
#include "TestMacros.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
void ReplicaGeoTest::test_trace() const
{
    {
        SCOPED_TRACE("Center +z");
        auto result = test_->track({0, 0.5, -990}, {0, 0, 1});
        GenericGeoTrackingResult ref;
        ref.volumes = {
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
        ref.volume_instances = {
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
        ref.distances = {
            190,
            149.5,
            1,
            48.5,
            0.99,
            0.019999999999991,
            0.99000000000001,
            48,
            0.99,
            0.019999999999991,
            0.99000000000001,
            48,
            0.99,
            0.019999999999991,
            0.99000000000001,
            48,
            0.99,
            0.019999999999991,
            0.99000000000001,
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
            0.023094010767563,
            1.1431535329954,
            110.17016486681,
            600,
        };
        ref.halfway_safeties = {
            95,
            74.75,
            0.5,
            24.25,
            0.495,
            0.01,
            0.495,
            24,
            0.495,
            0.01,
            0.495,
            24,
            0.495,
            0.01,
            0.495,
            24,
            0.495,
            0.01,
            0.495,
            24,
            0.495,
            0.01,
            0.495,
            99.5,
            50,
            99.5,
            31.698729810778,
            49.5,
            0.49499999999996,
            0.0099999999999922,
            0.49499999999997,
            22.457458783298,
            150,
        };
        ref.bumps = {};
        auto tol = GenericGeoTrackingTolerance::from_test(*test_);
        tol.distance *= 10;  // 2e-12 diff between g4, vg
        EXPECT_RESULT_NEAR(ref, result, tol);
    }
    {
        SCOPED_TRACE("Second arm");
        Real3 dir{0, 0, 0};
        sincos(Turn{-30.0 / 360.}, &dir[0], &dir[2]);
        auto result = test_->track({0.125, 0.5, 0.0625}, dir);

        GenericGeoTrackingResult ref;
        ref.volumes = {
            "magnetic",     "world",       "secondArm",    "chamber2",
            "wirePlane2",   "chamber2",    "secondArm",    "chamber2",
            "wirePlane2",   "chamber2",    "secondArm",    "chamber2",
            "wirePlane2",   "chamber2",    "secondArm",    "chamber2",
            "wirePlane2",   "chamber2",    "secondArm",    "chamber2",
            "wirePlane2",   "chamber2",    "secondArm",    "hodoscope2",
            "secondArm",    "cell",        "secondArm",    "HadCalLayer",
            "HadCalScinti", "HadCalLayer", "HadCalScinti", "HadCalLayer",
            "HadCalScinti", "HadCalLayer", "HadCalScinti", "HadCalLayer",
            "HadCalScinti", "HadCalLayer", "HadCalScinti", "HadCalLayer",
            "HadCalScinti", "HadCalLayer", "HadCalScinti", "HadCalLayer",
            "HadCalScinti", "HadCalLayer", "HadCalScinti", "HadCalLayer",
            "HadCalScinti", "HadCalLayer", "HadCalScinti", "HadCalLayer",
            "HadCalScinti", "HadCalLayer", "HadCalScinti", "HadCalLayer",
            "HadCalScinti", "HadCalLayer", "HadCalScinti", "HadCalLayer",
            "HadCalScinti", "HadCalLayer", "HadCalScinti", "HadCalLayer",
            "HadCalScinti", "HadCalLayer", "HadCalScinti", "world",
        };
        ref.volume_instances = {
            "magnetic",          "world_PV",          "fSecondArmPhys",
            "chamber2",          "wirePlane2",        "chamber2",
            "fSecondArmPhys",    "chamber2",          "wirePlane2",
            "chamber2",          "fSecondArmPhys",    "chamber2",
            "wirePlane2",        "chamber2",          "fSecondArmPhys",
            "chamber2",          "wirePlane2",        "chamber2",
            "fSecondArmPhys",    "chamber2",          "wirePlane2",
            "chamber2",          "fSecondArmPhys",    "hodoscope2",
            "fSecondArmPhys",    "cell_param@42",     "fSecondArmPhys",
            "HadCalLayer_PV@0",  "HadCalScinti",      "HadCalLayer_PV@1",
            "HadCalScinti",      "HadCalLayer_PV@2",  "HadCalScinti",
            "HadCalLayer_PV@3",  "HadCalScinti",      "HadCalLayer_PV@4",
            "HadCalScinti",      "HadCalLayer_PV@5",  "HadCalScinti",
            "HadCalLayer_PV@6",  "HadCalScinti",      "HadCalLayer_PV@7",
            "HadCalScinti",      "HadCalLayer_PV@8",  "HadCalScinti",
            "HadCalLayer_PV@9",  "HadCalScinti",      "HadCalLayer_PV@10",
            "HadCalScinti",      "HadCalLayer_PV@11", "HadCalScinti",
            "HadCalLayer_PV@12", "HadCalScinti",      "HadCalLayer_PV@13",
            "HadCalScinti",      "HadCalLayer_PV@14", "HadCalScinti",
            "HadCalLayer_PV@15", "HadCalScinti",      "HadCalLayer_PV@16",
            "HadCalScinti",      "HadCalLayer_PV@17", "HadCalScinti",
            "HadCalLayer_PV@18", "HadCalScinti",      "HadCalLayer_PV@19",
            "HadCalScinti",      "world_PV",
        };
        ref.distances = {
            100.00827610654,
            50.000097305727,
            99,
            0.98999999999997,
            0.020000000000008,
            0.99,
            48,
            0.99000000000001,
            0.019999999999994,
            0.99000000000001,
            48,
            0.99000000000001,
            0.019999999999994,
            0.99000000000001,
            48,
            0.98999999999995,
            0.019999999999994,
            0.99000000000001,
            48,
            0.99000000000001,
            0.019999999999994,
            0.99000000000001,
            48.5,
            1,
            184.5,
            30,
            35,
            4,
            0.99999999999999,
            4,
            0.99999999999999,
            4.0000000000001,
            0.99999999999999,
            4.0000000000001,
            0.99999999999999,
            4.0000000000001,
            0.99999999999999,
            4.0000000000001,
            0.99999999999999,
            4.0000000000001,
            0.99999999999999,
            4,
            0.99999999999999,
            4.0000000000001,
            0.99999999999999,
            4.0000000000001,
            0.99999999999999,
            4.0000000000001,
            0.99999999999999,
            4.0000000000001,
            0.99999999999999,
            4.0000000000001,
            0.99999999999999,
            4,
            0.99999999999999,
            4.0000000000001,
            0.99999999999999,
            4.0000000000001,
            0.99999999999999,
            4.0000000000001,
            0.99999999999999,
            4.0000000000001,
            0.99999999999999,
            4.0000000000001,
            0.99999999999999,
            4,
            0.99999999999999,
            304.61999618334,
        };
        ref.halfway_safeties = {
            50.004040731528,
            25.000029191686,
            49.5,
            0.49499999999999,
            0.0099999999999814,
            0.49499999999999,
            24,
            0.49499999999997,
            0.0099999999999798,
            0.49499999999997,
            24,
            0.49499999999997,
            0.0099999999999798,
            0.49499999999997,
            24,
            0.49499999999998,
            0.0099999999999798,
            0.49499999999997,
            24,
            0.49499999999997,
            0.0099999999999798,
            0.49499999999997,
            24.25,
            0.49999999999999,
            92.25,
            0.13950317547316,
            17.5,
            0.13950317547311,
            0.13950317547314,
            0.13950317547311,
            0.13950317547314,
            0.13950317547311,
            0.13950317547314,
            0.13950317547311,
            0.13950317547314,
            0.13950317547311,
            0.13950317547314,
            0.13950317547311,
            0.13950317547314,
            0.13950317547311,
            0.13950317547314,
            0.13950317547311,
            0.13950317547314,
            0.13950317547311,
            0.13950317547314,
            0.13950317547311,
            0.13950317547314,
            0.13950317547311,
            0.13950317547314,
            0.13950317547311,
            0.13950317547314,
            0.13950317547311,
            0.13950317547314,
            0.13950317547311,
            0.13950317547314,
            0.13950317547311,
            0.13950317547314,
            0.13950317547311,
            0.13950317547314,
            0.13950317547311,
            0.13950317547314,
            0.13950317547311,
            0.13950317547314,
            0.13950317547311,
            0.13950317547314,
            0.13950317547311,
            0.13950317547314,
            131.90432759775,
        };
        ref.bumps = {};
        auto tol = GenericGeoTrackingTolerance::from_test(*test_);
        tol.distance *= 10;  // 2e-12 diff between g4, vg
        EXPECT_RESULT_NEAR(ref, result, tol);
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
        // Geant4 gets stuck here (it's close to a boundary)
        auto result = test_->volume_stack({-342.5, 0.1, 593.22740159234});
        GenericGeoVolumeStackResult ref;
        ref.volume_instances = {
            "world_PV",
            "fSecondArmPhys",
        };
        ref.replicas = {-1, -1};
        if (test_->geometry_type() == "Geant4"
            || (test_->geometry_type() == "VecGeom"
                && CELERITAS_VECGEOM_SURFACE))
        {
            ref.volume_instances.insert(ref.volume_instances.end(),
                                        {"EMcalorimeter", "cell_param"});
            ref.replicas.insert(ref.replicas.end(), {-1, 42});
        }
        EXPECT_RESULT_EQ(ref, result);
    }
    {
        // A bit further along from the stuck point
        auto result = test_->volume_stack({-343, 0.1, 596});
        GenericGeoVolumeStackResult ref;
        ref.volume_instances = {
            "world_PV",
            "fSecondArmPhys",
            "EMcalorimeter",
            "cell_param",
        };
        ref.replicas = {-1, -1, -1, 42};
        EXPECT_RESULT_EQ(ref, result);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
