//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GeoTrackView.test.cc
//---------------------------------------------------------------------------//
#include "geometry/GeoTrackView.hh"

#include <VecGeom/navigation/NavigationState.h>
#include "geometry/GeoParams.hh"
#include "geometry/GeoStateStore.hh"
#include "comm/Device.hh"
#include "comm/Logger.hh"

#include "GeoParamsTest.hh"
#ifdef CELERITAS_USE_CUDA
#    include "GeoTrackView.test.hh"
#endif

#include "base/ArrayIO.hh"

using namespace celeritas;
using namespace celeritas_test;

//---------------------------------------------------------------------------//
// HOST TESTS
//---------------------------------------------------------------------------//

class GeoTrackViewHostTest : public GeoParamsTest
{
  public:
    using NavState = vecgeom::cxx::NavigationState;

    void SetUp() override
    {
        int max_depth = this->params()->max_depth();
        state.reset(NavState::MakeInstance(max_depth));
        next_state.reset(NavState::MakeInstance(max_depth));

        state_view.size       = 1;
        state_view.vgmaxdepth = max_depth;
        state_view.pos        = &this->pos;
        state_view.dir        = &this->dir;
        state_view.next_step  = &this->next_step;
        state_view.vgstate    = this->state.get();
        state_view.vgnext     = this->next_state.get();

        params_view = this->params()->host_pointers();
        CELER_ASSERT(params_view.world_volume);
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
    GeoParamsPointers params_view;
};

TEST_F(GeoTrackViewHostTest, track_line)
{
    // Construct geometry interface from persistent geometry data, state view,
    // and thread ID (which for CPU is just zero).
    GeoTrackView geo(params_view, state_view, ThreadId(0));

    {
        // Track from outside detector, moving right
        geo = {{-10, -10, -10}, {1, 0, 0}};
        EXPECT_EQ(VolumeId{0}, geo.volume_id()); // Shape2 center

        geo.find_next_step();
        EXPECT_SOFT_EQ(5, geo.next_step());
        geo.move_next_step();
        EXPECT_SOFT_EQ(-5, geo.pos()[0]);
        EXPECT_EQ(VolumeId{1}, geo.volume_id()); // Shape2 -> Shape1

        geo.find_next_step();
        EXPECT_SOFT_EQ(1, geo.next_step());
        geo.move_next_step();
        EXPECT_EQ(VolumeId{9}, geo.volume_id()); // Shape1 -> Envelope
        EXPECT_EQ(false, geo.is_outside());

        geo.find_next_step();
        EXPECT_SOFT_EQ(1, geo.next_step());
        geo.move_next_step();
        EXPECT_EQ(false, geo.is_outside()); // leaving World
    }

    {
        // Track from outside edge fails
        CELER_LOG(info) << "Init a track with a pointer outside work "
	                   "volume...";
        geo = {{24, 0, 0}, {-1, 0, 0}};
        EXPECT_EQ(geo.is_outside(), true);
    }

    {
        // But it works when you move juuust inside
        geo = {{-24 + 1e-3, 6.5, 6.5}, {1, 0, 0}};
        EXPECT_EQ(false, geo.is_outside());
        EXPECT_EQ(VolumeId{10}, geo.volume_id()); // World
        geo.find_next_step();
        EXPECT_SOFT_EQ(7. - 1e-3, geo.next_step());
        geo.move_next_step();
        EXPECT_EQ(VolumeId{3}, geo.volume_id()); // World -> Envelope
    }
    {
        // Track from inside detector
        geo = {{-10, 10, 10}, {0, -1, 0}};
        EXPECT_EQ(VolumeId{0}, geo.volume_id()); // Shape1 center

        geo.find_next_step();
        EXPECT_SOFT_EQ(5.0, geo.next_step());
        geo.move_next_step();
        EXPECT_SOFT_EQ(5.0, geo.pos()[1]);
        EXPECT_EQ(VolumeId{1}, geo.volume_id()); // Shape1 -> Shape2
        EXPECT_EQ(false, geo.is_outside());

        geo.find_next_step();
        EXPECT_SOFT_EQ(1.0, geo.next_step());
        geo.move_next_step();
        EXPECT_EQ(false, geo.is_outside());
    }
}

#if CELERITAS_USE_CUDA
//---------------------------------------------------------------------------//
// DEVICE TESTS
//---------------------------------------------------------------------------//

class GeoTrackViewDeviceTest : public GeoParamsTest
{
};

TEST_F(GeoTrackViewDeviceTest, track_lines)
{
    if (!celeritas::device())
    {
        SKIP("CUDA is disabled");
    }

    CELER_ASSERT(this->params());

    // Set up test input
    VGGTestInput input;
    input.init = {
        {{10, 10, 10}, {1, 0, 0}},
        {{10, 10, -10}, {1, 0, 0}},
        {{10, -10, 10}, {1, 0, 0}},
        {{10, -10, -10}, {1, 0, 0}},
        {{-10, 10, 10}, {-1, 0, 0}},
        {{-10, 10, -10}, {-1, 0, 0}},
        {{-10, -10, 10}, {-1, 0, 0}},
        {{-10, -10, -10}, {-1, 0, 0}}
    };
    input.max_segments = 3;
    input.shared       = this->params()->device_pointers();

    GeoStateStore device_states(*this->params(), input.init.size());
    input.state = device_states.device_pointers();

    // Run kernel
    auto output = vgg_test(input);

    static const int expected_ids[] = {
        0, 1, 2, 0, 1, 5, 0, 1, 4, 0, 1, 8,
        0, 1, 3, 0, 1, 7, 0, 1, 6, 0, 1, 9};

    static const double expected_distances[]
        = {5, 1, 1, 5, 1, 1, 5, 1, 1, 5, 1, 1,
           5, 1, 1, 5, 1, 1, 5, 1, 1, 5, 1, 1};

    // Check results
    EXPECT_VEC_EQ(expected_ids, output.ids);
    EXPECT_VEC_SOFT_EQ(output.distances, expected_distances);
}

//---------------------------------------------------------------------------//
#endif
