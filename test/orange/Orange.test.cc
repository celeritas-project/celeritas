//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Orange.test.cc
//---------------------------------------------------------------------------//
#include "orange/OrangeParams.hh"
#include "orange/OrangeTrackView.hh"

#include "base/Constants.hh"

#include "celeritas_test.hh"
#include "OrangeGeoTestBase.hh"
// #include "Orange.test.hh"

using namespace celeritas;
using namespace celeritas_test;

using celeritas::constants::sqrt_two;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class OrangeTest : public OrangeGeoTestBase
{
  protected:
    using Initializer_t = GeoTrackInitializer;

    void SetUp() override {}

    //! Create a host track view
    OrangeTrackView make_track_view()
    {
        if (!host_state_)
        {
            host_state_ = HostStateStore(this->params(), 1);
        }

        return OrangeTrackView(
            this->params().host_ref(), host_state_.ref(), ThreadId{0});
    }

  private:
    using HostStateStore
        = CollectionStateStore<OrangeStateData, MemSpace::host>;
    HostStateStore host_state_;
};

class OneVolumeTest : public OrangeTest
{
    void SetUp() override
    {
        OneVolInput geo_inp;
        this->build_geometry(geo_inp);
    }
};

class TwoVolumeTest : public OrangeTest
{
    void SetUp() override
    {
        TwoVolInput geo_inp;
        geo_inp.radius = 1.5;
        this->build_geometry(geo_inp);
    }
};

#define FiveVolumesTest TEST_IF_CELERITAS_JSON(FiveVolumesTest)
class FiveVolumesTest : public OrangeTest
{
    void SetUp() override { this->build_geometry("five-volumes.org.json"); }
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(OneVolumeTest, params)
{
    const OrangeParams& geo = this->params();

    EXPECT_EQ(1, geo.num_volumes());
    EXPECT_EQ(0, geo.num_surfaces());

    EXPECT_EQ("infinite", geo.id_to_label(VolumeId{0}));
    EXPECT_EQ(VolumeId{0}, geo.find_volume("infinite"));
}

TEST_F(OneVolumeTest, track_view)
{
    OrangeTrackView geo = this->make_track_view();

    // Initialize
    geo = Initializer_t{{3, 4, 5}, {1, 0, 0}};
    EXPECT_VEC_SOFT_EQ(Real3({3, 4, 5}), geo.pos());
    EXPECT_VEC_SOFT_EQ(Real3({1, 0, 0}), geo.dir());
    EXPECT_EQ(VolumeId{0}, geo.volume_id());
    EXPECT_EQ(SurfaceId{}, geo.surface_id());
    EXPECT_TRUE(geo.is_outside());

    // Try a boundary
    EXPECT_SOFT_EQ(inf, geo.find_next_step());
    geo.move_internal(2.5);
    EXPECT_VEC_SOFT_EQ(Real3({5.5, 4, 5}), geo.pos());

    // Move within the volume but not along a straight line
    geo.move_internal({5.6, 4.1, 5.1});
    EXPECT_VEC_SOFT_EQ(Real3({5.6, 4.1, 5.1}), geo.pos());

    // Change direction
    geo.set_dir({0, 1, 0});
    EXPECT_SOFT_EQ(inf, geo.find_next_step());
}

//---------------------------------------------------------------------------//

TEST_F(TwoVolumeTest, simple_track)
{
    auto geo = this->make_track_view();

    // Initialize
    geo = Initializer_t{{0.5, 0, 0}, {0, 0, 1}};
    EXPECT_VEC_SOFT_EQ(Real3({0.5, 0, 0}), geo.pos());
    EXPECT_VEC_SOFT_EQ(Real3({0, 0, 1}), geo.dir());
    EXPECT_EQ(VolumeId{1}, geo.volume_id());
    EXPECT_EQ(SurfaceId{}, geo.surface_id());
    EXPECT_FALSE(geo.is_outside());

    // Try a boundary; second call should be cached
    EXPECT_SOFT_EQ(sqrt_two, geo.find_next_step());
    EXPECT_SOFT_EQ(sqrt_two, geo.find_next_step());

    // Advance toward the boundary
    geo.move_internal(1);
    EXPECT_VEC_SOFT_EQ(Real3({0.5, 0, 1}), geo.pos());

    // Cross boundary
    geo.move_across_boundary();
    EXPECT_VEC_SOFT_EQ(Real3({0.5, 0, sqrt_two}), geo.pos());
    EXPECT_EQ(VolumeId{0}, geo.volume_id());
    EXPECT_EQ(SurfaceId{0}, geo.surface_id());
    EXPECT_TRUE(geo.is_outside());
}

TEST_F(TwoVolumeTest, persistence)
{
    {
        auto geo = this->make_track_view();
        geo      = Initializer_t{{2.5, 0, 0}, {-1, 0, 0}};
        geo.find_next_step();
        geo.move_across_boundary();
    }
    {
        auto geo = this->make_track_view();
        EXPECT_EQ(VolumeId{1}, geo.volume_id());
        EXPECT_EQ(SurfaceId{0}, geo.surface_id());
        EXPECT_VEC_SOFT_EQ(Real3({1.5, 0, 0}), geo.pos());
        EXPECT_VEC_SOFT_EQ(Real3({-1, 0, 0}), geo.dir());
        EXPECT_SOFT_EQ(3.0, geo.find_next_step());
        geo.move_across_boundary();
        EXPECT_EQ(VolumeId{0}, geo.volume_id());
        EXPECT_EQ(SurfaceId{0}, geo.surface_id());
        EXPECT_VEC_SOFT_EQ(Real3({-1.5, 0, 0}), geo.pos());
    }
}
