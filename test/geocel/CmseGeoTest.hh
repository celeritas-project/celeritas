//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/CmseGeoTest.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string_view>

#include "GenericGeoTestInterface.hh"
#include "TestMacros.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Test the transformed box geometry.
 */
class CmseGeoTest
{
  public:
    static std::string_view geometry_basename() { return "cmse"; }

    //! Construct with a reference to the GoogleTest
    CmseGeoTest(GenericGeoTestInterface* geo_test) : test_{geo_test} {}

    void test_accessors() const;
    inline void test_trace() const;

  private:
    GenericGeoTestInterface* test_;
};

//---------------------------------------------------------------------------//
void CmseGeoTest::test_trace() const
{
    // clang-format off
    {
        SCOPED_TRACE("Center +z");
        auto result = test_->track({0, 0, -4000}, {0, 0, 1});
        static char const* const expected_volumes[] = {"CMStoZDC", "BEAM3",
            "BEAM2", "BEAM1", "BEAM", "BEAM", "BEAM1", "BEAM2", "BEAM3",
            "CMStoZDC", "CMSE", "ZDC", "CMSE", "ZDCtoFP420", "CMSE"};
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static real_type const expected_distances[] = {1300, 1096.95, 549.15,
            403.9, 650, 650, 403.9, 549.15, 1096.95, 11200, 9.9999999999992,
            180, 910, 24000, 6000};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
        static real_type const expected_hw_safety[] = {100, 2.1499999999997,
            10.3027302206744, 13.023518051922, 6.95, 6.95, 13.023518051922,
            10.3027302206745, 2.15, 100, 5, 8, 100, 100, 100};
        EXPECT_VEC_SOFT_EQ(expected_hw_safety, result.halfway_safeties);
    }
    {
        SCOPED_TRACE("Offset +z");
        auto result = test_->track({30, 30, -4000}, {0, 0, 1});
        static char const* const expected_volumes[] = {"CMStoZDC", "OQUA",
            "VCAL", "OQUA", "CMSE", "TotemT1", "CMSE", "MUON", "CALO",
            "Tracker", "CALO", "MUON", "CMSE", "TotemT1", "CMSE", "OQUA",
            "VCAL", "OQUA", "CMStoZDC", "CMSE", "ZDCtoFP420", "CMSE"};
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static real_type const expected_distances[] = {1300, 1419.95, 165.1,
            28.95, 36, 300.1, 94.858988388759, 100.94101161124, 260.9, 586.4,
            260.9, 100.94101161124, 94.858988388759, 300.1, 36, 28.95, 165.1,
            1419.95, 11200, 1100, 24000, 6000};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
        static real_type const expected_hw_safety[] = {57.573593128807,
            40.276406871193, 29.931406871193, 14.475, 18, 28.702447147997,
            29.363145173005, 32.665765921596, 34.260814069425, 39.926406871193,
            34.260814069425, 32.665765921596, 29.363145173005, 28.702447147997,
            18, 14.475, 29.931406871193, 40.276406871193, 57.573593128807,
            57.573593128807, 57.573593128807, 57.573593128807};
        EXPECT_VEC_SOFT_EQ(expected_hw_safety, result.halfway_safeties);
    }
    {
        SCOPED_TRACE("Across muon");
        auto result = test_->track({-1000, 0, -48.5}, {1, 0, 0});
        static char const* const expected_volumes[] = {"OCMS", "MUON", "CALO",
            "Tracker", "CMSE", "BEAM", "CMSE", "Tracker", "CALO", "MUON",
            "OCMS"};
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static real_type const expected_distances[] = {170, 535, 171.7, 120.8,
            0.15673306650246, 4.6865338669951, 0.15673306650246, 120.8, 171.7,
            535, 920};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
        static real_type const expected_hw_safety[] = {85, 267.5, 85.85,
            60.4, 0.078366388350241, 2.343262600759, 0.078366388350241,
            60.4, 85.85, 267.5, 460};
        EXPECT_VEC_SOFT_EQ(expected_hw_safety, result.halfway_safeties);
    }
    {
        SCOPED_TRACE("Differs between G4/VG");
        auto result = test_->track({0, 0, 1328.0}, {1, 0, 0});
        static char const* const expected_volumes[] = {"BEAM2", "OQUA", "CMSE",
            "OCMS"};
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static real_type const expected_distances[] = {12.495, 287.505, 530,
            920};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
        static real_type const expected_hw_safety[] = {6.2475, 47.95, 242, 460};
        EXPECT_VEC_SOFT_EQ(expected_hw_safety, result.halfway_safeties);
    }
    // clang-format on
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
