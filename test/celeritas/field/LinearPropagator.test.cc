//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/LinearPropagator.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/field/LinearPropagator.hh"

#include <string_view>

#include "corecel/cont/ArrayIO.hh"
#include "corecel/data/CollectionStateStore.hh"
#include "corecel/io/Logger.hh"
#include "corecel/sys/Device.hh"
#include "geocel/UnitUtils.hh"
#include "celeritas/CoreGeoTestBase.hh"
#include "celeritas/geo/CoreGeoTrackView.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class LinearPropagatorTest : public CoreGeoTestBase
{
  protected:
    std::string_view gdml_basename() const final { return "simple-cms"; }
};

//---------------------------------------------------------------------------//
// HOST TESTS
//----------------------------------------------------------------------------//

TEST_F(LinearPropagatorTest, rvalue_type)
{
    using GeoTrackView = WrappedGeoTrack;
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

TEST_F(LinearPropagatorTest, simple_cms)
{
    using GeoTrackView = WrappedGeoTrack;
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
        Propagation result = propagate(numeric_limits<real_type>::infinity());
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
