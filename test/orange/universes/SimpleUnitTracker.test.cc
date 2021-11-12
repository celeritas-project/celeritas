//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SimpleUnitTracker.test.cc
//---------------------------------------------------------------------------//
#include "orange/universes/SimpleUnitTracker.hh"

// Source includes
#include "base/ArrayUtils.hh"
#include "base/Constants.hh"

// Test includes
#include "celeritas_test.hh"
#include "orange/OrangeGeoTestBase.hh"
// #include "SimpleUnitTracker.test.hh"

using namespace celeritas;
using celeritas::constants::sqrt_two;
using celeritas::detail::Initialization;
using celeritas::detail::LocalState;

namespace
{
constexpr real_type sqrt_half = sqrt_two / 2;
}

//---------------------------------------------------------------------------//
// TEST HARNESSES
//---------------------------------------------------------------------------//

class SimpleUnitTrackerTest : public celeritas_test::OrangeGeoTestBase
{
  protected:
    using LocalState = SimpleUnitTracker::LocalState;

    // Initialization without any logical state
    LocalState make_state(Real3 pos, Real3 dir)
    {
        normalize_direction(&dir);
        LocalState state;
        state.pos         = pos;
        state.dir         = dir;
        state.volume      = {};
        state.surface     = {};
        state.temp_senses = this->sense_storage();
        return state;
    }

    // Initialization crossing a surface with *before crossing volume*
    // and *before crossing sense*
    LocalState make_state(
        Real3 pos, Real3 dir, const char* vol, const char* surf, char sense)
    {
        Sense before_crossing_sense;
        switch (sense)
        {
            case '-':
                before_crossing_sense = Sense::inside;
                break;
            case '+':
                before_crossing_sense = Sense::outside;
                break;
            default:
                CELER_VALIDATE(false,
                               << "invalid sense value '" << sense << "'");
        }

        LocalState state = this->make_state(pos, dir);
        state.volume     = this->find_volume(vol);
        // *Intentionally* flip the sense because we're looking for the
        // post-crossing volume. This is normally done by the multi-level
        // TrackingGeometry.
        state.surface
            = {this->find_surface(surf), flip_sense(before_crossing_sense)};
        return state;
    }
};

class OneVolumeTest : public SimpleUnitTrackerTest
{
    void SetUp() override
    {
        OneVolInput geo_inp;
        this->build_geometry(geo_inp);
    }
};

class TwoVolumeTest : public SimpleUnitTrackerTest
{
    void SetUp() override
    {
        TwoVolInput geo_inp;
        geo_inp.radius = 1.5;
        this->build_geometry(geo_inp);
    }
};

//! Construct a test name that is disabled when JSON is disabled
#if CELERITAS_USE_JSON
#    define TEST_IF_CELERITAS_JSON(name) name
#else
#    define TEST_IF_CELERITAS_JSON(name) DISABLED_##name
#endif

#define FiveVolumesTest TEST_IF_CELERITAS_JSON(FiveVolumesTest)
class FiveVolumesTest : public SimpleUnitTrackerTest
{
    void SetUp() override
    {
        if (!CELERITAS_USE_JSON)
        {
            GTEST_SKIP() << "JSON is not enabled";
        }

        this->build_geometry("five-volumes.org.json");
    }
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(OneVolumeTest, initialize)
{
    SimpleUnitTracker tracker(this->params_host_ref());

    {
        // Anywhere is valid
        auto init = tracker.initialize(this->make_state({1, 2, 3}, {0, 0, 1}));
        EXPECT_TRUE(init);
        EXPECT_EQ(VolumeId{0}, init.volume);
        EXPECT_FALSE(init.surface);
    }
}

//---------------------------------------------------------------------------//

TEST_F(TwoVolumeTest, initialize)
{
    SimpleUnitTracker tracker(this->params_host_ref());

    {
        SCOPED_TRACE("In the inner sphere");
        auto init
            = tracker.initialize(this->make_state({0.5, 0, 0}, {0, 0, 1}));
        EXPECT_EQ("inside", this->id_to_label(init.volume));
        EXPECT_FALSE(init.surface);
    }
    {
        // Test rejection
        SCOPED_TRACE("On the boundary but not crossing a surface");
        auto init
            = tracker.initialize(this->make_state({1.5, 0, 0}, {0, 0, 1}));
        EXPECT_FALSE(init);
    }
    {
        SCOPED_TRACE("Crossing the boundary from the inside");
        auto init = tracker.initialize(
            this->make_state({1.5, 0, 0}, {0, 0, 1}, "inside", "sphere", '-'));
        EXPECT_EQ("outside", this->id_to_label(init.volume));
        EXPECT_EQ("sphere", this->id_to_label(init.surface.id()));
        EXPECT_EQ(Sense::outside, init.surface.sense());
    }
    {
        SCOPED_TRACE("Crossing the boundary from the outside");
        auto init = tracker.initialize(this->make_state(
            {1.5, 0, 0}, {0, 0, 1}, "outside", "sphere", '+'));
        EXPECT_EQ("inside", this->id_to_label(init.volume));
        EXPECT_EQ("sphere", this->id_to_label(init.surface.id()));
        EXPECT_EQ(Sense::inside, init.surface.sense());
    }
    {
        SCOPED_TRACE("Outside the sphere");
        auto init
            = tracker.initialize(this->make_state({3.0, 0, 0}, {0, 0, 1}));
        EXPECT_EQ("outside", this->id_to_label(init.volume));
        EXPECT_FALSE(init.surface);
    }
}

//---------------------------------------------------------------------------//

TEST_F(FiveVolumesTest, initialize)
{
    SimpleUnitTracker tracker(this->params_host_ref());

    {
        SCOPED_TRACE("Exterior");
        auto init = tracker.initialize(
            this->make_state({1000, 1000, -1000}, {1, 0, 0}));
        EXPECT_EQ("[EXTERIOR]", this->id_to_label(init.volume));
        EXPECT_FALSE(init.surface);
    }
    {
        SCOPED_TRACE("Single sphere 'e'");
        auto init
            = tracker.initialize(this->make_state({-.25, -.25, 0}, {0, 0, 1}));
        EXPECT_EQ("e", this->id_to_label(init.volume));
        EXPECT_FALSE(init.surface);
    }
    {
        SCOPED_TRACE("Trimmed square 'a'");
        auto init
            = tracker.initialize(this->make_state({-.7, .7, 0}, {0, 0, 1}));
        EXPECT_EQ("a", this->id_to_label(init.volume));
        EXPECT_FALSE(init.surface);
    }
    {
        SCOPED_TRACE("Complicated fill volume 'd'");
        auto init
            = tracker.initialize(this->make_state({.75, 0.2, 0}, {0, 0, 1}));
        EXPECT_EQ("d", this->id_to_label(init.volume));
        EXPECT_FALSE(init.surface);
    }
    {
        // Triple point between a, c, d
        SCOPED_TRACE("On the boundary but not crossing a surface");
        auto init
            = tracker.initialize(this->make_state({0, 0.75, 0}, {1, 1, 0}));
        EXPECT_FALSE(init);
    }
    {
        SCOPED_TRACE("Crossing the boundary from the inside of 'e'");
        auto init = tracker.initialize(this->make_state(
            {-0.5, -0.25, 0}, {-1, 0, 0}, "e", "epsilon.s", '-'));
        EXPECT_EQ("c", this->id_to_label(init.volume));
        EXPECT_EQ("epsilon.s", this->id_to_label(init.surface.id()));
        EXPECT_EQ(Sense::outside, init.surface.sense());
    }
    {
        SCOPED_TRACE(
            "Crossing the boundary from the inside of 'e' but with "
            "numerical imprecision");
        real_type eps  = 1e-10;
        auto      init = tracker.initialize(this->make_state(
            {eps, -0.25, 0}, {1, 0, 0}, "e", "epsilon.s", '-'));
        EXPECT_EQ("c", this->id_to_label(init.volume));
        EXPECT_EQ("epsilon.s", this->id_to_label(init.surface.id()));
        EXPECT_EQ(Sense::outside, init.surface.sense());
    }
    {
        SCOPED_TRACE("Crossing the boundary into a more complicated region");
        auto init = tracker.initialize(
            this->make_state({-.75 * sqrt_half, 0.75 * sqrt_half, 0},
                             {0, 1, 0},
                             "c",
                             "gamma.s",
                             '-'));
        EXPECT_EQ("a", this->id_to_label(init.volume));
        EXPECT_EQ("gamma.s", this->id_to_label(init.surface.id()));
        EXPECT_EQ(Sense::outside, init.surface.sense());
    }
    {
        SCOPED_TRACE("Crossing back in from a complicated region");
        auto init = tracker.initialize(
            this->make_state({-.75 * sqrt_half, 0.75 * sqrt_half, 0},
                             {0, -1, 0},
                             "a",
                             "gamma.s",
                             '+'));
        EXPECT_EQ("c", this->id_to_label(init.volume));
        EXPECT_EQ("gamma.s", this->id_to_label(init.surface.id()));
        EXPECT_EQ(Sense::inside, init.surface.sense());
    }
    {
        // TODO: this might not be OK since intersection logic may not be
        // correct when exactly on a boundary but not *known* to be on that
        // boundary. We'll either need to ensure that's ei
        SCOPED_TRACE("Crossing at triple point");
        auto init = tracker.initialize(
            this->make_state({0, 0.75, 0}, {0, 1, 0}, "c", "gamma.s", '-'));
        EXPECT_EQ("d", this->id_to_label(init.volume));
        EXPECT_EQ("gamma.s", this->id_to_label(init.surface.id()));
        EXPECT_EQ(Sense::outside, init.surface.sense());

        init = tracker.initialize(
            this->make_state({0, 0.75, 0}, {-1, 0, 0}, "d", "gamma.s", '+'));
        EXPECT_EQ("c", this->id_to_label(init.volume));
        EXPECT_EQ("gamma.s", this->id_to_label(init.surface.id()));
        EXPECT_EQ(Sense::inside, init.surface.sense());

        // Near triple point, on sphere but crossing plane edge
        init = tracker.initialize(
            this->make_state({0, 0.75, 0}, {-1, 0, 0}, "d", "alpha.px", '+'));
        EXPECT_EQ("a", this->id_to_label(init.volume));
        EXPECT_EQ("alpha.px", this->id_to_label(init.surface.id()));
        EXPECT_EQ(Sense::inside, init.surface.sense());
    }
}
