//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LinearPropagator.test.cc
//---------------------------------------------------------------------------//
#include "geometry/LinearPropagator.hh"

#include "base/ArrayIO.hh"
#include "base/CollectionStateStore.hh"
#include "comm/Device.hh"
#include "comm/Logger.hh"
#include "geometry/GeoParams.hh"
#include "geometry/GeoData.hh"

#include "celeritas_test.hh"
#include "GeoTestBase.hh"

using namespace celeritas;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class LinearPropagatorTest
    : public celeritas_test::GeoTestBase<celeritas::GeoParams>
{
  public:
    using StateStore = CollectionStateStore<GeoStateData, MemSpace::host>;

    const char* dirname() const override { return "geometry"; }
    const char* filebase() const override { return "simple-cms"; }

    void SetUp() override { state = StateStore(*this->geometry(), 1); }

    GeoTrackView make_geo_track_view()
    {
        return GeoTrackView(
            this->geometry()->host_ref(), state.ref(), ThreadId{0});
    }

    std::string volume_label(const GeoTrackView& geo)
    {
        if (geo.is_outside())
        {
            return "[OUTSIDE]";
        }
        return this->geometry()->id_to_label(geo.volume_id());
    }

  protected:
    StateStore state;
};

//---------------------------------------------------------------------------//
// HOST TESTS
//----------------------------------------------------------------------------//

TEST_F(LinearPropagatorTest, all)
{
    GeoTrackView geo = this->make_geo_track_view();

    // Initialize
    geo = {{0, 0, 0}, {0, 0, 1}};
    EXPECT_EQ("vacuum_tube", this->volume_label(geo));

    {
        LinearPropagator propagate(&geo);

        // Move up to a small distance
        Propagation result = propagate(20);
        EXPECT_SOFT_EQ(20, result.distance);
        EXPECT_FALSE(result.boundary);
    }

    // Check state and scatter
    EXPECT_VEC_SOFT_EQ(Real3({0, 0, 20}), geo.pos());
    EXPECT_EQ("vacuum_tube", this->volume_label(geo));
    geo.set_dir({1, 0, 0});

    {
        LinearPropagator propagate(&geo);

        // Move to the next layer
        Propagation result = propagate(1e20);
        EXPECT_SOFT_EQ(30, result.distance);
        EXPECT_TRUE(result.boundary);
        geo.cross_boundary();
    }

    // Check state
    EXPECT_VEC_SOFT_EQ(Real3({30, 0, 20}), geo.pos());
    EXPECT_EQ("si_tracker", this->volume_label(geo));

    {
        LinearPropagator propagate(&geo);

        // Move two steps internally
        Propagation result = propagate(35);
        EXPECT_SOFT_EQ(35, result.distance);
        EXPECT_FALSE(result.boundary);

        result = propagate(40);
        EXPECT_SOFT_EQ(40, result.distance);
        EXPECT_FALSE(result.boundary);
    }

    // Check state
    EXPECT_VEC_SOFT_EQ(Real3({105, 0, 20}), geo.pos());
    EXPECT_EQ("si_tracker", this->volume_label(geo));

    {
        LinearPropagator propagate(&geo);

        // Move to next boundary (infinite max distance)
        Propagation result = propagate();
        EXPECT_SOFT_EQ(20, result.distance);
        EXPECT_TRUE(result.boundary);
        geo.cross_boundary();

        // Move slightly inside before next scatter
        result = propagate(0.1);
        EXPECT_SOFT_EQ(0.1, result.distance);
        EXPECT_FALSE(result.boundary);
    }

    // Check state and scatter
    EXPECT_VEC_SOFT_EQ(Real3({125.1, 0, 20}), geo.pos());
    EXPECT_EQ("em_calorimeter", this->volume_label(geo));
    geo.set_dir({0, 0, -1});

    {
        LinearPropagator propagate(&geo);

        // Move to world volume
        Propagation result = propagate(10000);
        EXPECT_SOFT_EQ(720, result.distance);
        EXPECT_TRUE(result.boundary);
        geo.cross_boundary();

        // Move outside
        result = propagate(10000);
        EXPECT_SOFT_EQ(1300, result.distance);
        EXPECT_TRUE(result.boundary);
        geo.cross_boundary();
    }

    EXPECT_VEC_SOFT_EQ(Real3({125.1, 0, -2000}), geo.pos());
    EXPECT_EQ("[OUTSIDE]", this->volume_label(geo));
}
