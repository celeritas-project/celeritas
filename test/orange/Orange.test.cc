//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/Orange.test.cc
//---------------------------------------------------------------------------//
#include "orange/OrangeParams.hh"
#include "orange/OrangeTrackView.hh"
#include "celeritas/Constants.hh"

#include "OrangeGeoTestBase.hh"
#include "celeritas_test.hh"
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

#define Geant4Testem15Test TEST_IF_CELERITAS_JSON(Geant4Testem15Test)
class Geant4Testem15Test : public OrangeTest
{
    void SetUp() override { this->build_geometry("geant4-testem15.org.json"); }
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(OneVolumeTest, params)
{
    const OrangeParams& geo = this->params();

    EXPECT_EQ(1, geo.num_volumes());
    EXPECT_EQ(0, geo.num_surfaces());
    EXPECT_TRUE(geo.supports_safety());

    EXPECT_EQ("infinite", geo.id_to_label(VolumeId{0}).name);
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
    auto next = geo.find_next_step();
    EXPECT_SOFT_EQ(inf, next.distance);
    EXPECT_FALSE(next.boundary);
    geo.move_internal(2.5);
    EXPECT_VEC_SOFT_EQ(Real3({5.5, 4, 5}), geo.pos());

    // Move within the volume but not along a straight line
    geo.move_internal({5.6, 4.1, 5.1});
    EXPECT_VEC_SOFT_EQ(Real3({5.6, 4.1, 5.1}), geo.pos());

    // Change direction
    geo.set_dir({0, 1, 0});
    next = geo.find_next_step();
    EXPECT_SOFT_EQ(inf, next.distance);
    EXPECT_FALSE(next.boundary);

    // Get safety distance
    EXPECT_SOFT_EQ(inf, geo.find_safety());
}

//---------------------------------------------------------------------------//

TEST_F(TwoVolumeTest, params)
{
    const OrangeParams& geo = this->params();

    EXPECT_EQ(2, geo.num_volumes());
    EXPECT_EQ(1, geo.num_surfaces());
    EXPECT_TRUE(geo.supports_safety());

    EXPECT_EQ("sphere", geo.id_to_label(SurfaceId{0}).name);
    EXPECT_EQ(SurfaceId{0}, geo.find_surface("sphere"));

    const auto& bbox = geo.bbox();
    EXPECT_VEC_SOFT_EQ((Real3{-1.5, -1.5, -1.5}), bbox.lower());
    EXPECT_VEC_SOFT_EQ((Real3{1.5, 1.5, 1.5}), bbox.upper());
}

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
    auto next = geo.find_next_step();
    EXPECT_SOFT_EQ(sqrt_two, next.distance);
    EXPECT_TRUE(next.boundary);
    next = geo.find_next_step();
    EXPECT_SOFT_EQ(sqrt_two, next.distance);
    EXPECT_TRUE(next.boundary);

    // Advance toward the boundary
    geo.move_internal(1);
    EXPECT_VEC_SOFT_EQ(Real3({0.5, 0, 1}), geo.pos());
    EXPECT_EQ(SurfaceId{}, geo.surface_id());
    // Next step should still be cached
    next = geo.find_next_step();
    EXPECT_SOFT_EQ(sqrt_two - 1, next.distance);
    EXPECT_TRUE(next.boundary);

    // Move to boundary
    geo.move_to_boundary();
    EXPECT_VEC_SOFT_EQ(Real3({0.5, 0, sqrt_two}), geo.pos());
    EXPECT_EQ(VolumeId{1}, geo.volume_id());
    EXPECT_EQ(SurfaceId{0}, geo.surface_id());
    EXPECT_FALSE(geo.is_outside());
    EXPECT_DOUBLE_EQ(0.0, geo.find_safety());

    // Logically flip the surface into the new volume
    geo.cross_boundary();
    EXPECT_EQ(VolumeId{0}, geo.volume_id());
    EXPECT_EQ(SurfaceId{0}, geo.surface_id());
    EXPECT_TRUE(geo.is_outside());
    EXPECT_DOUBLE_EQ(0.0, geo.find_safety());

    // Move internally to an arbitrary position
    geo.find_next_step();
    geo.move_internal({2, 2, 0});
    EXPECT_EQ(SurfaceId{}, geo.surface_id());
    geo.set_dir({0, 1, 0});
    EXPECT_SOFT_EQ(2 * sqrt_two - 1.5, geo.find_safety());
    geo.set_dir({-sqrt_two / 2, -sqrt_two / 2, 0});

    next = geo.find_next_step();
    EXPECT_SOFT_EQ(2 * sqrt_two - 1.5, next.distance);
    EXPECT_TRUE(next.boundary);
    geo.move_to_boundary();
    geo.cross_boundary();
    EXPECT_EQ(VolumeId{1}, geo.volume_id());
    EXPECT_EQ(SurfaceId{0}, geo.surface_id());
}

TEST_F(TwoVolumeTest, persistence)
{
    {
        auto geo = this->make_track_view();
        geo      = Initializer_t{{2.5, 0, 0}, {-1, 0, 0}};
        geo.find_next_step();
        geo.move_to_boundary();
    }
    {
        auto geo = this->make_track_view();
        EXPECT_EQ(VolumeId{0}, geo.volume_id());
        EXPECT_EQ(SurfaceId{0}, geo.surface_id());
        EXPECT_VEC_SOFT_EQ(Real3({1.5, 0, 0}), geo.pos());
        EXPECT_VEC_SOFT_EQ(Real3({-1, 0, 0}), geo.dir());
        geo.cross_boundary();
    }
    {
        auto geo = this->make_track_view();
        EXPECT_EQ(VolumeId{1}, geo.volume_id());
        EXPECT_EQ(SurfaceId{0}, geo.surface_id());
        EXPECT_VEC_SOFT_EQ(Real3({1.5, 0, 0}), geo.pos());
        EXPECT_VEC_SOFT_EQ(Real3({-1, 0, 0}), geo.dir());
        auto next = geo.find_next_step();
        EXPECT_SOFT_EQ(3.0, next.distance);
        EXPECT_TRUE(next.boundary);
        geo.move_to_boundary();
        geo.cross_boundary();
        EXPECT_EQ(VolumeId{0}, geo.volume_id());
        EXPECT_EQ(SurfaceId{0}, geo.surface_id());
        EXPECT_VEC_SOFT_EQ(Real3({-1.5, 0, 0}), geo.pos());
    }
    {
        auto geo = this->make_track_view();
        EXPECT_EQ(VolumeId{0}, geo.volume_id());
        EXPECT_EQ(SurfaceId{0}, geo.surface_id());
        EXPECT_VEC_SOFT_EQ(Real3({-1.5, 0, 0}), geo.pos());
        geo.move_internal({-1.5, .5, .5});
        EXPECT_EQ(SurfaceId{}, geo.surface_id());
    }
    {
        auto geo = this->make_track_view();
        EXPECT_VEC_SOFT_EQ(Real3({-1.5, .5, .5}), geo.pos());
        geo.set_dir({1, 0, 0});
    }
    {
        auto geo = this->make_track_view();
        EXPECT_VEC_SOFT_EQ(Real3({1, 0, 0}), geo.dir());
        auto next = geo.find_next_step();
        EXPECT_SOFT_EQ(0.17712434446770464, next.distance);
        EXPECT_TRUE(next.boundary);
        geo.move_internal(0.1);
        EXPECT_EQ(SurfaceId{}, geo.surface_id());
    }
    {
        auto geo = this->make_track_view();
        EXPECT_VEC_SOFT_EQ(Real3({-1.4, .5, .5}), geo.pos());
        EXPECT_EQ(SurfaceId{}, geo.surface_id());
        auto next = geo.find_next_step();
        EXPECT_SOFT_EQ(0.07712434446770464, next.distance);
        EXPECT_TRUE(next.boundary);
    }
}

TEST_F(TwoVolumeTest, intersect_limited)
{
    auto geo = this->make_track_view();

    // Initialize
    geo = Initializer_t{{0.0, 0, 0}, {1, 0, 0}};

    // Try a boundary; second call should be cached
    auto next = geo.find_next_step(0.5);
    EXPECT_SOFT_EQ(0.5, next.distance);
    EXPECT_FALSE(next.boundary);
    next = geo.find_next_step(0.5);
    EXPECT_SOFT_EQ(0.5, next.distance);
    EXPECT_FALSE(next.boundary);
    if (CELERITAS_DEBUG)
    {
        EXPECT_THROW(geo.move_to_boundary(), celeritas::DebugError);
    }

    // Move almost to that point, nearby step should be the same
    geo.move_internal(0.45);
    EXPECT_VEC_SOFT_EQ(Real3({0.45, 0, 0}), geo.pos());
    next = geo.find_next_step(0.05);
    EXPECT_SOFT_EQ(0.05, next.distance);
    EXPECT_FALSE(next.boundary);

    // Find the next step further away
    next = geo.find_next_step(2.0);
    EXPECT_SOFT_EQ(1.05, next.distance);
    EXPECT_TRUE(next.boundary);
    geo.move_to_boundary();
    EXPECT_VEC_SOFT_EQ(Real3({1.5, 0, 0}), geo.pos());
    EXPECT_EQ(VolumeId{1}, geo.volume_id());
    EXPECT_EQ(SurfaceId{0}, geo.surface_id());

    geo.cross_boundary();
    EXPECT_EQ(VolumeId{0}, geo.volume_id());
    for (real_type d : {10, 5, 20})
    {
        next = geo.find_next_step(d);
        EXPECT_SOFT_EQ(d, next.distance);
        EXPECT_FALSE(next.boundary);
    }
    next = geo.find_next_step();
    EXPECT_SOFT_EQ(inf, next.distance);
    EXPECT_FALSE(next.boundary);
    next = geo.find_next_step(12345.0);
    EXPECT_SOFT_EQ(12345.0, next.distance);
    EXPECT_FALSE(next.boundary);
}

TEST_F(FiveVolumesTest, params)
{
    const OrangeParams& geo = this->params();

    EXPECT_EQ(6, geo.num_volumes());
    EXPECT_EQ(12, geo.num_surfaces());
    EXPECT_FALSE(geo.supports_safety());
}

TEST_F(Geant4Testem15Test, params)
{
    const OrangeParams& geo = this->params();

    EXPECT_EQ(3, geo.num_volumes());
    EXPECT_EQ(12, geo.num_surfaces());
    // The 'world' volume includes a negated box
    EXPECT_FALSE(geo.supports_safety());
}

TEST_F(Geant4Testem15Test, safety)
{
    OrangeTrackView geo = this->make_track_view();

    geo = Initializer_t{{0, 0, 0}, {1, 0, 0}};
    EXPECT_VEC_SOFT_EQ(Real3({0, 0, 0}), geo.pos());
    EXPECT_VEC_SOFT_EQ(Real3({1, 0, 0}), geo.dir());
    EXPECT_EQ(VolumeId{1}, geo.volume_id());
    EXPECT_EQ(SurfaceId{}, geo.surface_id());
    EXPECT_FALSE(geo.is_outside());

    // Safety at middle should be to the box boundary
    EXPECT_SOFT_EQ(5000.0, geo.find_safety());

    // Check safety near face
    auto next = geo.find_next_step(4995.0);
    EXPECT_SOFT_EQ(4995.0, next.distance);
    EXPECT_FALSE(next.boundary);
    geo.move_internal(4995.0);
    EXPECT_SOFT_EQ(5.0, geo.find_safety());

    // Check safety near edge
    geo.set_dir({0, 1, 0});
    next = geo.find_next_step();
    geo.move_internal(4990.0);
    EXPECT_SOFT_EQ(5.0, geo.find_safety());
    geo.set_dir({-1, 0, 0});
    next = geo.find_next_step();
    geo.move_internal(6.0);
    EXPECT_SOFT_EQ(10.0, geo.find_safety());
}
