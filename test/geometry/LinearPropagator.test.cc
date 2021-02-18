//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LinearPropagator.test.cc
//---------------------------------------------------------------------------//
#include "geometry/LinearPropagator.hh"

#include "geometry/GeoStateStore.hh"

#include "GeoParamsTest.hh"
#ifdef CELERITAS_USE_CUDA
#    include "LinearPropagator.test.hh"
#endif

#include "base/ArrayIO.hh"
#include "comm/Device.hh"

using namespace celeritas;
using namespace celeritas_test;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class LinearPropagatorHostTest : public GeoParamsTest
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

//---------------------------------------------------------------------------//
// HOST TESTS
//---------------------------------------------------------------------------//

TEST_F(LinearPropagatorHostTest, accessors)
{
    const auto& geom = *params();
    EXPECT_EQ(11, geom.num_volumes());
    EXPECT_EQ(4, geom.max_depth());
    EXPECT_EQ("Shape2", geom.id_to_label(VolumeId{0}));
    EXPECT_EQ("Shape1", geom.id_to_label(VolumeId{1}));
}

//----------------------------------------------------------------------------//

TEST_F(LinearPropagatorHostTest, track_line)
{
    GeoTrackView     geo(params_view, state_view, ThreadId(0));
    LinearPropagator propagate(&geo); // one propagator per track

    {
        // Track from outside detector, moving right
        geo = {{-10, 10, 10}, {1, 0, 0}};
        EXPECT_EQ(VolumeId{0}, geo.volume_id()); // Shape2 center

        geo.find_next_step();
        EXPECT_SOFT_EQ(5, geo.next_step());
        propagate();
        EXPECT_SOFT_EQ(-5, geo.pos()[0]);
        EXPECT_EQ(VolumeId{1}, geo.volume_id()); // Shape2 -> Shape1

        geo.find_next_step();
        EXPECT_SOFT_EQ(1.0, geo.next_step());
        propagate();
        EXPECT_EQ(VolumeId{3}, geo.volume_id()); // Shape1 -> Envelope
        EXPECT_EQ(false, geo.is_outside());

        geo.find_next_step();
        EXPECT_SOFT_EQ(1.0, geo.next_step());
        propagate();
        EXPECT_EQ(false, geo.is_outside()); // leaving World
    }

    {
        // Track from outside edge fails
        geo = {{24, 0, 0}, {-1, 0, 0}};
        EXPECT_EQ(true, geo.is_outside());
    }

    {
        // But it works when you move juuust inside
        real_type eps = 1e-6;
        geo           = {{-24 + eps, 6.5, 6.5}, {1, 0, 0}};
        EXPECT_EQ(false, geo.is_outside());
        EXPECT_EQ(VolumeId{10}, geo.volume_id()); // World
        geo.find_next_step();
        EXPECT_SOFT_EQ(7. - eps, geo.next_step());
        propagate();
        EXPECT_EQ(VolumeId{3}, geo.volume_id()); // World -> Envelope
    }
    {
        // Track from inside detector
        geo = {{-10, 10, 10}, {0, -1, 0}};
        EXPECT_EQ(VolumeId{0}, geo.volume_id()); // Shape1 center

        geo.find_next_step();
        EXPECT_SOFT_EQ(5.0, geo.next_step());
        propagate();
        EXPECT_SOFT_EQ(5.0, geo.pos()[1]);
        EXPECT_EQ(VolumeId{1}, geo.volume_id()); // Shape1 -> Shape2
        EXPECT_EQ(false, geo.is_outside());

        geo.find_next_step();
        EXPECT_SOFT_EQ(1.0, geo.next_step());
        propagate();
        EXPECT_EQ(false, geo.is_outside());
    }
}

//----------------------------------------------------------------------------//

TEST_F(LinearPropagatorHostTest, track_intraVolume)
{
    GeoTrackView     geo(params_view, state_view, ThreadId(0));
    LinearPropagator propagate(&geo); // one propagator per track

    {
        // Track from outside detector, moving right
        geo = {{-10, 10, 10}, {0, 0, 1}};
        EXPECT_EQ(VolumeId{0}, geo.volume_id()); // Shape2

        geo.find_next_step();
        EXPECT_SOFT_EQ(5, geo.next_step());

        // break next step into two
        propagate(0.5 * geo.next_step());
        EXPECT_SOFT_EQ(12.5, geo.pos()[2]);
        EXPECT_EQ(VolumeId{0}, geo.volume_id()); // still Shape2

        propagate(geo.next_step()); // all remaining
        EXPECT_SOFT_EQ(15.0, geo.pos()[2]);
        EXPECT_EQ(VolumeId{1}, geo.volume_id()); // Shape2 -> Shape1

        // break next step into > 2 steps, re-calculating next_step each time
        geo.find_next_step();
        EXPECT_SOFT_EQ(1.0, geo.next_step()); // dist to next boundary

        propagate(0.2 * geo.next_step()); // step 1 inside Detector
        EXPECT_SOFT_EQ(15.2, geo.pos()[2]);
        EXPECT_EQ(VolumeId{1}, geo.volume_id()); // Shape1

        geo.find_next_step();
        EXPECT_SOFT_EQ(0.8, geo.next_step());

        propagate(0.5 * geo.next_step()); // step 2 inside Detector
        EXPECT_SOFT_EQ(15.6, geo.pos()[2]);
        EXPECT_EQ(VolumeId{1}, geo.volume_id()); // Shape1

        geo.find_next_step();
        EXPECT_SOFT_EQ(0.4, geo.next_step());

        propagate(geo.next_step()); // last step inside Detector
        EXPECT_SOFT_EQ(16, geo.pos()[2]);
        EXPECT_EQ(VolumeId{3}, geo.volume_id()); // Shape1 -> Envelope
    }
}

#if CELERITAS_USE_CUDA
//---------------------------------------------------------------------------//
// DEVICE TESTS
//---------------------------------------------------------------------------//

class LinearPropagatorDeviceTest : public GeoParamsTest
{
};

TEST_F(LinearPropagatorDeviceTest, track_lines)
{
    if (!celeritas::device())
    {
        SKIP("CUDA is disabled");
    }

    CELER_ASSERT(this->params());

    // Set up test input
    LinPropTestInput input;
    input.init = {
        {{10, 10, 10}, {-1, 0, 0}},
        {{10, -10, -10}, {0, 1, 0}},
        {{-10, 10, -10}, {0, 0, 1}},
        {{0, 0, 0}, {1, 1, 1}},
    };
    input.max_segments = 3;
    input.shared       = this->params()->device_pointers();

    GeoStateStore device_states(*this->params(), input.init.size());
    input.state = device_states.device_pointers();

    // Run kernel
    auto output = linProp_test(input);

    static const int expected_ids[] = {0, 1, 2, 0, 1, 8, 0, 1, 7, 10, 2, 1};
    static const double expected_distances[]
        = {5, 1, 1, 5, 1, 2, 5, 1, 3, 3, 1, 2.47582530373998 };

    // Check results
    EXPECT_VEC_EQ(expected_ids, output.ids);
    EXPECT_VEC_SOFT_EQ(output.distances, expected_distances);
}

//---------------------------------------------------------------------------//
#endif
