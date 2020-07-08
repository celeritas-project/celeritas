//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file VGGeometry.test.cc
//---------------------------------------------------------------------------//
#include "geometry/VGGeometry.hh"

#include <memory>
#include <VecGeom/navigation/NavigationState.h>
#include "gtest/Main.hh"
#include "gtest/Test.hh"
#include "geometry/VGHost.hh"
#include "celeritas_config.h"
// #include "VGGeometry.test.hh"

using namespace celeritas;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class VGGeometryTest : public celeritas::Test
{
  protected:
    using constSPVGHost = std::shared_ptr<const VGHost>;

    static void SetUpTestCase()
    {
        std::string test_file = std::string(CELERITAS_SOURCE_DIR)
                                + "/test/geometry/data/twoBoxes.gdml";
        host_geom_ = std::make_shared<VGHost>(test_file.c_str());
    }

    static void TearDownTestCase() { host_geom_.reset(); }

    const constSPVGHost& host_geom()
    {
        ENSURE(host_geom_);
        return host_geom_;
    }

  private:
    static constSPVGHost host_geom_;
};

VGGeometryTest::constSPVGHost VGGeometryTest::host_geom_ = nullptr;

//---------------------------------------------------------------------------//
// HOST TESTS
//---------------------------------------------------------------------------//

class VGGeometryHostTest : public VGGeometryTest
{
  public:
    using NavState = vecgeom::cxx::NavigationState;

    void SetUp() override
    {
        int max_depth = host_geom()->max_depth();
        state.reset(NavState::MakeInstance(max_depth));
        next_state.reset(NavState::MakeInstance(max_depth));

        state_view.size       = 1;
        state_view.vgmaxdepth = max_depth;
        state_view.pos        = &this->pos;
        state_view.dir        = &this->dir;
        state_view.next_step  = &this->next_step;
        state_view.vgstate    = this->state.get();
        state_view.vgnext     = this->next_state.get();

        host_view = host_geom()->host_view();
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
    VGStateView state_view;
    VGView      host_view;
};

TEST_F(VGGeometryHostTest, track_line)
{
    // Construct geometry interface from persistent geometry data, state view,
    // and thread ID (which for CPU is just zero).
    VGGeometry geo(host_view, state_view, ThreadId(0));

    geo.construct({-6, 0, 0}, {1, 0, 0});
    EXPECT_EQ(VolumeId{1}, geo.volume_id()); // World
    EXPECT_EQ(Boundary::inside, geo.boundary());

    geo.find_next_step();
    EXPECT_SOFT_EQ(1.0, geo.next_step());
    geo.move_next_step();
    EXPECT_SOFT_EQ(-5.0 + VGGeometry::step_fudge(), geo.pos()[0]);
    EXPECT_EQ(VolumeId{0}, geo.volume_id()); // Detector

    geo.find_next_step();
    EXPECT_SOFT_EQ(10.0 - VGGeometry::step_fudge(), geo.next_step());
    geo.move_next_step();
    EXPECT_EQ(VolumeId{1}, geo.volume_id()); // World

    geo.find_next_step();
    EXPECT_SOFT_EQ(45.0 - VGGeometry::step_fudge(), geo.next_step());
    geo.move_next_step();
    EXPECT_EQ(Boundary::outside, geo.boundary());

    geo.destroy();
}
