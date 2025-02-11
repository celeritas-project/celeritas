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
    : geo_test_{geo_test}
{
    CELER_EXPECT(geo_test_);
}

//---------------------------------------------------------------------------//
//! Test geometry accessors
void MultiLevelGeoTest::test_accessors() const
{
    auto const& geo = *geo_test_->geometry_interface();
    EXPECT_EQ(3, geo.max_depth());

    static char const* const expected_vol_names[] = {
        "sph",
        "box",
        "world",
    };
    EXPECT_VEC_EQ(expected_vol_names, geo_test_->get_volume_names());

    static char const* const expected_vol_inst_names[] = {
        "boxsph1",
        "boxsph2",
        "topsph1",
        "topbox1",
        "topbox2",
        "topbox3",
        "topbox4",
        "world_PV",
    };
    EXPECT_VEC_EQ(expected_vol_inst_names,
                  geo_test_->get_volume_instance_names());
}

//---------------------------------------------------------------------------//
void MultiLevelGeoTest::test_trace() const
{
    {
        SCOPED_TRACE("high");
        auto result = geo_test_->track({-19.9, 7.5, 0}, {1, 0, 0});
        result.print_expected();
    }
    {
        SCOPED_TRACE("low");
        auto result = geo_test_->track({-19.9, -7.5, 0}, {1, 0, 0});
        result.print_expected();
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
