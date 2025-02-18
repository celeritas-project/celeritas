//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/MultiLevelGeoTest.cc
//---------------------------------------------------------------------------//
#include "MultiLevelGeoTest.hh"

#include "TestMacros.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Construct with tracking test interface.
 */
MultiLevelGeoTest::MultiLevelGeoTest(GenericGeoTestInterface* geo_test)
    : test_{geo_test}
{
    CELER_EXPECT(test_);
}

//---------------------------------------------------------------------------//
//! Test geometry accessors
void MultiLevelGeoTest::test_accessors() const
{
    auto const& geo = *test_->geometry_interface();
    EXPECT_EQ(3, geo.max_depth());

    static char const* const expected_vol_labels[] = {
        "sph",
        "tri",
        "box",
        "world",
        "box_refl",
        "sph_refl",
        "tri_refl",
    };
    EXPECT_VEC_EQ(expected_vol_labels, test_->get_volume_labels());

    static char const* const expected_vol_inst_labels[] = {
        "boxsph1@0",
        "boxsph2@0",
        "boxtri@0",
        "topbox1",
        "topsph1",
        "topbox2",
        "topbox3",
        "boxsph1@1",
        "boxsph2@1",
        "boxtri@1",
        "topbox4",
        "world_PV",
    };
    EXPECT_VEC_EQ(expected_vol_inst_labels,
                  test_->get_volume_instance_labels());

    if (test_->g4world())
    {
        EXPECT_VEC_EQ(expected_vol_inst_labels, test_->get_g4pv_labels());
    }
}

//---------------------------------------------------------------------------//
void MultiLevelGeoTest::test_trace() const
{
    {
        SCOPED_TRACE("high");
        auto result = test_->track({-19.9, 7.5, 0}, {1, 0, 0});
        static char const* const expected_volumes[] = {
            "world",
            "box",
            "sph",
            "box",
            "tri",
            "box",
            "world",
            "box",
            "sph",
            "box",
            "tri",
            "box",
            "world",
        };
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static char const* const expected_volume_instances[] = {
            "world_PV",
            "topbox2",
            "boxsph2",
            "topbox2",
            "boxtri",
            "topbox2",
            "world_PV",
            "topbox1",
            "boxsph2",
            "topbox1",
            "boxtri",
            "topbox1",
            "world_PV",
        };
        EXPECT_VEC_EQ(expected_volume_instances, result.volume_instances);
        static real_type const expected_distances[] = {
            2.4,
            3,
            4,
            1.8452994616207,
            2.3094010767585,
            3.8452994616207,
            5,
            3,
            4,
            1.8452994616207,
            2.3094010767585,
            3.8452994616207,
            6.5,
        };
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
        static real_type const expected_hw_safety[] = {
            1.2,
            1.5,
            2,
            0.79903810567666,
            1,
            1.6650635094611,
            2.5,
            1.5,
            2,
            0.79903810567666,
            1,
            1.6650635094611,
            3.25,
        };
        EXPECT_VEC_SOFT_EQ(expected_hw_safety, result.halfway_safeties);
    }
    {
        SCOPED_TRACE("low");
        auto result = test_->track({-19.9, -7.5, 0}, {1, 0, 0});
        static char const* const expected_volumes[] = {
            "world",
            "box",
            "sph",
            "box",
            "world",
            "box_refl",
            "sph_refl",
            "box_refl",
            "tri_refl",
            "box_refl",
            "world",
        };
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static char const* const expected_volume_instances[] = {
            "world_PV",
            "topbox3",
            "boxsph2",
            "topbox3",
            "world_PV",
            "topbox4",
            "boxsph2",
            "topbox4",
            "boxtri",
            "topbox4",
            "world_PV",
        };
        EXPECT_VEC_EQ(expected_volume_instances, result.volume_instances);
        static real_type const expected_distances[] = {
            2.4,
            3,
            4,
            8,
            5,
            3,
            4,
            1.8452994616207,
            2.3094010767585,
            3.8452994616207,
            6.5,
        };
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
        static real_type const expected_hw_safety[] = {
            1.2,
            1.5,
            2,
            3.0990195135928,
            2.5,
            1.5,
            2,
            0.79903810567666,
            1,
            1.6650635094611,
            3.25,
        };
        EXPECT_VEC_SOFT_EQ(expected_hw_safety, result.halfway_safeties);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
