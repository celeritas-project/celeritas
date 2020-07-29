//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GeoTrackView.test.cc
//---------------------------------------------------------------------------//
#include "geometry/GeoTrackView.hh"

#include <memory>
#include <VecGeom/navigation/NavigationState.h>
#include "gtest/Main.hh"
#include "gtest/Test.hh"
#include "celeritas_config.h"
#include "geometry/GeoParams.hh"
#include "geometry/GeoStateStore.hh"
#include "GeoTrackView.test.hh"

using namespace celeritas;
using namespace celeritas_test;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class GeoTrackViewTest : public celeritas::Test
{
  protected:
    using SptrConstParams = std::shared_ptr<const GeoParams>;

    static void SetUpTestCase()
    {
        std::string test_file = std::string(CELERITAS_SOURCE_DIR)
                                + "/test/geometry/data/twoBoxes.gdml";
        geom_ = std::make_shared<GeoParams>(test_file.c_str());
    }

    static void TearDownTestCase() { geom_.reset(); }

    const SptrConstParams& params()
    {
        ENSURE(geom_);
        return geom_;
    }

  private:
    static SptrConstParams geom_;
};

GeoTrackViewTest::SptrConstParams GeoTrackViewTest::geom_ = nullptr;

//---------------------------------------------------------------------------//
// HOST TESTS
//---------------------------------------------------------------------------//

class GeoTrackViewHostTest : public GeoTrackViewTest
{
  public:
    using NavState = vecgeom::cxx::NavigationState;

    void SetUp() override
    {
        int max_depth = params()->max_depth();
        state.reset(NavState::MakeInstance(max_depth));
        next_state.reset(NavState::MakeInstance(max_depth));

        state_view.size       = 1;
        state_view.vgmaxdepth = max_depth;
        state_view.pos        = &this->pos;
        state_view.dir        = &this->dir;
        state_view.next_step  = &this->next_step;
        state_view.vgstate    = this->state.get();
        state_view.vgnext     = this->next_state.get();

        host_view = params()->host_view();
        CHECK(host_view.world_volume);
    }

  protected:
    // State data
    Real3                     pos;
    Real3                     dir;
    real_type                 next_step;
    std::unique_ptr<NavState> state;
    std::unique_ptr<NavState> next_state;

    // Views
    GeoStatePointers  state_view;
    GeoParamsPointers host_view;
};

TEST_F(GeoTrackViewHostTest, accessors)
{
    const auto& geom = *params();
    EXPECT_EQ(2, geom.num_volumes());
    EXPECT_EQ(2, geom.max_depth());
    EXPECT_EQ("Detector", geom.id_to_label(VolumeId{0}));
    EXPECT_EQ("World", geom.id_to_label(VolumeId{1}));
}

TEST_F(GeoTrackViewHostTest, track_line)
{
    // Construct geometry interface from persistent geometry data, state view,
    // and thread ID (which for CPU is just zero).
    GeoTrackView geo(host_view, state_view, ThreadId(0));

    {
        // Track from outside detector, moving right
        geo = {{-6, 0, 0}, {1, 0, 0}};
        EXPECT_EQ(VolumeId{1}, geo.volume_id()); // World
        EXPECT_EQ(Boundary::inside, geo.boundary());

        geo.find_next_step();
        EXPECT_SOFT_EQ(1.0, geo.next_step());
        geo.move_next_step();
        EXPECT_SOFT_EQ(-5.0, geo.pos()[0]);
        EXPECT_EQ(VolumeId{0}, geo.volume_id()); // Detector

        geo.find_next_step();
        EXPECT_SOFT_EQ(10.0, geo.next_step());
        geo.move_next_step();
        EXPECT_EQ(VolumeId{1}, geo.volume_id()); // World

        geo.find_next_step();
        EXPECT_SOFT_EQ(45.0, geo.next_step());
        geo.move_next_step();
        EXPECT_EQ(Boundary::outside, geo.boundary());
    }

    {
        // Track from outside edge fails
        geo = {{50, 0, 0}, {-1, 0, 0}};
        EXPECT_EQ(Boundary::outside, geo.boundary());
    }

    {
        // But it works when you move juuust inside
        geo = {{50 - 1e-6, 0, 0}, {-1, 0, 0}};
        EXPECT_EQ(VolumeId{1}, geo.volume_id()); // World
        geo.find_next_step();
        EXPECT_SOFT_EQ(45.0 - 1e-6, geo.next_step());
        geo.move_next_step();
        EXPECT_EQ(VolumeId{0}, geo.volume_id()); // Detector
    }
    {
        // Track from inside detector
        geo = {{0, 0, 0}, {1, 0, 0}};
        EXPECT_EQ(VolumeId{0}, geo.volume_id()); // Detector
        EXPECT_EQ(Boundary::inside, geo.boundary());

        geo.find_next_step();
        EXPECT_SOFT_EQ(5.0, geo.next_step());
        geo.move_next_step();
        EXPECT_SOFT_EQ(5.0, geo.pos()[0]);
        EXPECT_EQ(VolumeId{1}, geo.volume_id()); // World

        geo.find_next_step();
        EXPECT_SOFT_EQ(45.0, geo.next_step());
        geo.move_next_step();
        EXPECT_EQ(Boundary::outside, geo.boundary());
    }
}

#if CELERITAS_USE_CUDA
//---------------------------------------------------------------------------//
// DEVICE TESTS
//---------------------------------------------------------------------------//

class GeoTrackViewDeviceTest : public GeoTrackViewTest
{
};

TEST_F(GeoTrackViewDeviceTest, track_lines)
{
    CHECK(params());

    // Set up test input
    VGGTestInput input;
    input.init = {
        {{-6, 0, 0}, {1, 0, 0}},
        {{0, 0, 0}, {1, 0, 0}},
        {{50, 0, 0}, {-1, 0, 0}},
        {{50 - 1e-6, 0, 0}, {-1, 0, 0}},
    };
    input.max_segments = 3;
    input.shared       = params()->device_pointers();

    GeoStateStore device_states(params(), input.init.size());
    input.state = device_states.device_pointers();

    // Run kernel
    auto output = vgg_test(input);

    static const int expected_ids[] = {1, 0, 1, 0, 1, -1, -1, -1, -1, 1, 0, 1};
    static const double expected_distances[]
        = {1, 10, 45, 5, 45, -1, -1, -1, -1, 45 - 1e-6, 10, 45};

    // Check results
    EXPECT_VEC_EQ(expected_ids, output.ids);
    EXPECT_VEC_SOFT_EQ(output.distances, expected_distances);
}

//---------------------------------------------------------------------------//
#endif
