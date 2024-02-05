//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/LinearPropagator.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/field/LinearPropagator.hh"

#include "corecel/cont/ArrayIO.hh"
#include "corecel/data/CollectionStateStore.hh"
#include "corecel/io/Logger.hh"
#include "corecel/sys/Device.hh"
#include "geocel/UnitUtils.hh"
#include "celeritas/AllGeoTypedTestBase.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

template<class HP>
class LinearPropagatorTest : public AllGeoTypedTestBase<HP>
{
  protected:
    using SPConstGeo = typename GenericGeoTestBase<HP>::SPConstGeo;
    using GeoTrackView = typename GenericGeoTestBase<HP>::GeoTrackView;

    void SetUp() override
    {
        if (CELERITAS_UNITS != CELERITAS_UNITS_CGS
            && this->geo_name() == "ORANGE")
        {
            GTEST_SKIP() << "ORANGE currently requires CGS [cm]";
        }
    }

    std::string geometry_basename() const final { return "simple-cms"; }
};

TYPED_TEST_SUITE(LinearPropagatorTest,
                 AllGeoTestingTypes,
                 AllGeoTestingTypeNames);

//---------------------------------------------------------------------------//
// HOST TESTS
//----------------------------------------------------------------------------//

TYPED_TEST(LinearPropagatorTest, rvalue_type)
{
    using GeoTrackView = typename TestFixture::GeoTrackView;
    {
        LinearPropagator propagate(
            this->make_geo_track_view({0, 0, 0}, {0, 0, 1}));
        EXPECT_TRUE((
            std::is_same_v<decltype(propagate), LinearPropagator<GeoTrackView>>));
        Propagation result = propagate(from_cm(10));
        EXPECT_SOFT_EQ(10, to_cm(result.distance));
        EXPECT_FALSE(result.boundary);
    }
    EXPECT_VEC_SOFT_EQ(Real3({0, 0, 10}),
                       to_cm(this->make_geo_track_view().pos()));
}

TYPED_TEST(LinearPropagatorTest, simple_cms)
{
    using GeoTrackView = typename TestFixture::GeoTrackView;
    // Initialize
    auto geo = this->make_geo_track_view({0, 0, 0}, {0, 0, 1});
    EXPECT_EQ("vacuum_tube", this->volume_name(geo));

    {
        LinearPropagator propagate(geo);
        EXPECT_TRUE((std::is_same_v<decltype(propagate),
                                    LinearPropagator<GeoTrackView&>>));

        // Move up to a small distance
        Propagation result = propagate(from_cm(20));
        EXPECT_SOFT_EQ(20, to_cm(result.distance));
        EXPECT_FALSE(result.boundary);
    }

    // Check state and scatter
    EXPECT_VEC_SOFT_EQ(Real3({0, 0, 20}), to_cm(geo.pos()));
    EXPECT_EQ("vacuum_tube", this->volume_name(geo));
    geo.set_dir({1, 0, 0});

    {
        LinearPropagator propagate(geo);

        // Move to the result layer
        Propagation result = propagate(from_cm(1e20));
        EXPECT_SOFT_EQ(30, to_cm(result.distance));
        EXPECT_TRUE(result.boundary);
        geo.cross_boundary();
    }

    // Check state
    EXPECT_VEC_SOFT_EQ(Real3({30, 0, 20}), to_cm(geo.pos()));
    EXPECT_EQ("si_tracker", this->volume_name(geo));

    {
        LinearPropagator propagate(geo);

        // Move two steps internally
        Propagation result = propagate(from_cm(35));
        EXPECT_SOFT_EQ(35, to_cm(result.distance));
        EXPECT_FALSE(result.boundary);

        result = propagate(from_cm(40));
        EXPECT_SOFT_EQ(40, to_cm(result.distance));
        EXPECT_FALSE(result.boundary);
    }

    // Check state
    EXPECT_VEC_SOFT_EQ(Real3({105, 0, 20}), to_cm(geo.pos()));
    EXPECT_EQ("si_tracker", this->volume_name(geo));

    {
        LinearPropagator propagate(geo);

        // Move to result boundary (infinite max distance)
        Propagation result = propagate();
        EXPECT_SOFT_EQ(20, to_cm(result.distance));
        EXPECT_TRUE(result.boundary);
        geo.cross_boundary();

        // Move slightly inside before result scatter
        result = propagate(from_cm(0.1));
        EXPECT_SOFT_EQ(0.1, to_cm(result.distance));
        EXPECT_FALSE(result.boundary);
    }

    // Check state and scatter
    EXPECT_VEC_SOFT_EQ(Real3({125.1, 0, 20}), to_cm(geo.pos()));
    EXPECT_EQ("em_calorimeter", this->volume_name(geo));
    geo.set_dir({0, 0, -1});

    {
        LinearPropagator propagate(geo);

        // Move to world volume
        Propagation result = propagate(from_cm(10000));
        EXPECT_SOFT_EQ(720, to_cm(result.distance));
        EXPECT_TRUE(result.boundary);
        geo.cross_boundary();

        // Move outside
        result = propagate(from_cm(10000));
        EXPECT_SOFT_EQ(1300, to_cm(result.distance));
        EXPECT_TRUE(result.boundary);
        geo.cross_boundary();
    }

    EXPECT_VEC_SOFT_EQ(Real3({125.1, 0, -2000}), to_cm(geo.pos()));
    EXPECT_EQ("[OUTSIDE]", this->volume_name(geo));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
