//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/Orange.test.cc
//---------------------------------------------------------------------------//
#include "orange/OrangeParams.hh"
#include "orange/OrangeTrackView.hh"
#include "orange/Types.hh"
#include "orange/construct/OrangeInput.hh"
#include "celeritas/Constants.hh"

#include "OrangeGeoTestBase.hh"
#include "TestMacros.hh"
#include "celeritas_test.hh"

using celeritas::constants::sqrt_two;

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class OrangeTest : public OrangeGeoTestBase
{
  protected:
    using Initializer_t = GeoTrackInitializer;

    //! Create a host track view
    OrangeTrackView make_track_view(TrackSlotId tsid = TrackSlotId{0})
    {
        if (!host_state_)
        {
            host_state_ = HostStateStore(this->host_params(), 2);
        }
        CELER_EXPECT(tsid < host_state_.size());

        return OrangeTrackView(this->host_params(), host_state_.ref(), tsid);
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

#define UniversesTest TEST_IF_CELERITAS_JSON(UniversesTest)
class UniversesTest : public OrangeTest
{
    void SetUp() override { this->build_geometry("universes.org.json"); }
};

#define RectArrayTest TEST_IF_CELERITAS_JSON(RectArrayTest)
class RectArrayTest : public OrangeTest
{
    void SetUp() override { this->build_geometry("rect_array.org.json"); }
};

#define Geant4Testem15Test TEST_IF_CELERITAS_JSON(Geant4Testem15Test)
class Geant4Testem15Test : public OrangeTest
{
    void SetUp() override { this->build_geometry("geant4-testem15.org.json"); }
};

//---------------------------------------------------------------------------//

TEST_F(OneVolumeTest, params)
{
    OrangeParams const& geo = this->params();

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
    geo = Initializer_t{{3, 4, 5}, {0, 1, 0}};
    EXPECT_VEC_SOFT_EQ(Real3({3, 4, 5}), geo.pos());
    EXPECT_VEC_SOFT_EQ(Real3({0, 1, 0}), geo.dir());
    EXPECT_EQ(VolumeId{0}, geo.volume_id());
    EXPECT_EQ(SurfaceId{}, geo.surface_id());
    EXPECT_TRUE(geo.is_outside());
    EXPECT_FALSE(geo.is_on_boundary());

    // Initialize from a pre-existing OrangeTrackView object
    geo = OrangeTrackView::DetailedInitializer{geo, Real3({1, 0, 0})};
    EXPECT_VEC_SOFT_EQ(Real3({3, 4, 5}), geo.pos());
    EXPECT_VEC_SOFT_EQ(Real3({1, 0, 0}), geo.dir());
    EXPECT_EQ(VolumeId{0}, geo.volume_id());
    EXPECT_EQ(SurfaceId{}, geo.surface_id());
    EXPECT_TRUE(geo.is_outside());
    EXPECT_FALSE(geo.is_on_boundary());

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
    OrangeParams const& geo = this->params();

    EXPECT_EQ(2, geo.num_volumes());
    EXPECT_EQ(1, geo.num_surfaces());
    EXPECT_TRUE(geo.supports_safety());

    EXPECT_EQ("sphere", geo.id_to_label(SurfaceId{0}).name);
    EXPECT_EQ(SurfaceId{0}, geo.find_surface("sphere"));

    auto const& bbox = geo.bbox();
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
    EXPECT_FALSE(geo.is_on_boundary());

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
    EXPECT_TRUE(geo.is_on_boundary());
    if (CELERITAS_DEBUG)
    {
        EXPECT_THROW(geo.find_safety(), celeritas::DebugError);
    }

    // Logically flip the surface into the new volume
    geo.cross_boundary();
    EXPECT_EQ(VolumeId{0}, geo.volume_id());
    EXPECT_EQ(SurfaceId{0}, geo.surface_id());
    EXPECT_TRUE(geo.is_outside());
    EXPECT_TRUE(geo.is_on_boundary());

    // Move internally to an arbitrary position
    geo.find_next_step();
    EXPECT_TRUE(geo.is_on_boundary());
    geo.move_internal({2, 2, 0});
    EXPECT_EQ(SurfaceId{}, geo.surface_id());
    EXPECT_FALSE(geo.is_on_boundary());
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

// Leaving the volume almost at a tangent, but magnetic field changes direction
// on boundary so it ends up heading back in
TEST_F(TwoVolumeTest, reentrant_boundary_setdir)
{
    auto geo = this->make_track_view();
    geo = Initializer_t{{1.49, 0, 0}, {0, 1, 0}};
    EXPECT_EQ(VolumeId{1}, geo.volume_id());
    EXPECT_EQ(SurfaceId{}, geo.surface_id());

    {
        // Find distance
        Propagation next = geo.find_next_step();
        EXPECT_TRUE(next.boundary);
        EXPECT_SOFT_EQ(0.17291616465790594, next.distance);
    }
    {
        // Move to boundary
        geo.move_to_boundary();
        EXPECT_VEC_SOFT_EQ(Real3({1.49, 0.172916164657906, 0}), geo.pos());
        EXPECT_EQ(VolumeId{1}, geo.volume_id());
        EXPECT_EQ(SurfaceId{0}, geo.surface_id());
    }
    {
        // Scatter on boundary so we're heading back into volume 1
        geo.set_dir({-1, 0, 0});
        EXPECT_EQ(VolumeId{1}, geo.volume_id());
        EXPECT_EQ(SurfaceId{0}, geo.surface_id());
    }
    {
        // Cross back into volume
        geo.cross_boundary();
        EXPECT_EQ(VolumeId{1}, geo.volume_id());
        EXPECT_EQ(SurfaceId{0}, geo.surface_id());
    }
    {
        // Find next distance
        Propagation next = geo.find_next_step();
        EXPECT_TRUE(next.boundary);
        EXPECT_SOFT_EQ(2.98, next.distance);
    }
}

TEST_F(TwoVolumeTest, nonreentrant_boundary_setdir)
{
    auto geo = this->make_track_view();
    geo = Initializer_t{{1.49, 0, 0}, {0, 1, 0}};
    EXPECT_EQ(VolumeId{1}, geo.volume_id());
    EXPECT_EQ(SurfaceId{}, geo.surface_id());

    {
        // Find distance
        Propagation next = geo.find_next_step();
        EXPECT_TRUE(next.boundary);
        EXPECT_SOFT_EQ(0.17291616465790594, next.distance);
    }
    {
        // Move to boundary
        geo.move_to_boundary();
        EXPECT_VEC_SOFT_EQ(Real3({1.49, 0.172916164657906, 0}), geo.pos());
        EXPECT_EQ(VolumeId{1}, geo.volume_id());
        EXPECT_EQ(SurfaceId{0}, geo.surface_id());
    }
    {
        // Scatter on boundary so we're still leaving volume 1
        geo.set_dir({1, 0, 0});
        EXPECT_EQ(VolumeId{1}, geo.volume_id());
        EXPECT_EQ(SurfaceId{0}, geo.surface_id());
    }
    {
        // Cross into new volume
        geo.cross_boundary();
        EXPECT_EQ(VolumeId{0}, geo.volume_id());
        EXPECT_EQ(SurfaceId{0}, geo.surface_id());
    }
}

// Leaving the voume almost at a tangent, but magnetic field changes direction
// on boundary so it ends up heading back in, then MSC changes it back outward
// again
TEST_F(TwoVolumeTest, doubly_reentrant_boundary_setdir)
{
    auto geo = this->make_track_view();
    geo = Initializer_t{{1.49, 0, 0}, {0, 1, 0}};
    EXPECT_EQ(VolumeId{1}, geo.volume_id());
    EXPECT_EQ(SurfaceId{}, geo.surface_id());

    {
        // Find distance
        Propagation next = geo.find_next_step();
        EXPECT_TRUE(next.boundary);
        EXPECT_SOFT_EQ(0.17291616465790594, next.distance);
    }
    {
        // Move to boundary
        geo.move_to_boundary();
        EXPECT_VEC_SOFT_EQ(Real3({1.49, 0.172916164657906, 0}), geo.pos());
        EXPECT_EQ(VolumeId{1}, geo.volume_id());
        EXPECT_EQ(SurfaceId{0}, geo.surface_id());
    }
    {
        // Scatter on boundary so we're heading back into volume 1
        geo.set_dir({-1, 0, 0});
        EXPECT_EQ(VolumeId{1}, geo.volume_id());
        EXPECT_EQ(SurfaceId{0}, geo.surface_id());
    }
    {
        // Scatter again so we're headed out
        geo.set_dir({1, 0, 0});
        EXPECT_EQ(VolumeId{1}, geo.volume_id());
        EXPECT_EQ(SurfaceId{0}, geo.surface_id());
    }
    {
        // Cross into new volume
        geo.cross_boundary();
        EXPECT_EQ(VolumeId{0}, geo.volume_id());
        EXPECT_EQ(SurfaceId{0}, geo.surface_id());
    }
}

// After leaving the volume almost at a tangent, change direction before moving
// as part of the field propagation algorithm.
TEST_F(TwoVolumeTest, reentrant_boundary_setdir_post)
{
    auto geo = this->make_track_view();
    geo = Initializer_t{{1.49, 0, 0}, {0, 1, 0}};
    EXPECT_EQ(VolumeId{1}, geo.volume_id());
    EXPECT_EQ(SurfaceId{}, geo.surface_id());

    {
        // Find distance
        Propagation next = geo.find_next_step();
        EXPECT_TRUE(next.boundary);
        EXPECT_SOFT_EQ(0.17291616465790594, next.distance);
    }
    {
        // Move to boundary
        geo.move_to_boundary();
        EXPECT_VEC_SOFT_EQ(Real3({1.49, 0.172916164657906, 0}), geo.pos());
        EXPECT_EQ(VolumeId{1}, geo.volume_id());
        EXPECT_EQ(SurfaceId{0}, geo.surface_id());

        // Cross into new volume
        geo.cross_boundary();
        EXPECT_EQ(VolumeId{0}, geo.volume_id());
        EXPECT_EQ(SurfaceId{0}, geo.surface_id());
    }
    {
        // Propose direction on boundary so we're heading back into volume 1
        EXPECT_TRUE(geo.is_on_boundary());
        geo.set_dir({-1, 0, 0});

        // Find distance
        Propagation next = geo.find_next_step();
        EXPECT_TRUE(next.boundary);
        EXPECT_SOFT_EQ(0, next.distance);

        // Propose a new direction but still headed back inside
        EXPECT_TRUE(geo.is_on_boundary());
        geo.set_dir({-sqrt_two / 2, sqrt_two / 2, 0});
        next = geo.find_next_step();
        EXPECT_TRUE(next.boundary);
        EXPECT_SOFT_EQ(0, next.distance);

        // Propose a new direction headed outside again
        EXPECT_TRUE(geo.is_on_boundary());
        geo.set_dir({0, 1, 0});
        next = geo.find_next_step();
        EXPECT_FALSE(next.boundary);
        EXPECT_SOFT_EQ(inf, next.distance);
    }
}

TEST_F(TwoVolumeTest, persistence)
{
    {
        auto geo = this->make_track_view();
        geo = Initializer_t{{2.5, 0, 0}, {-1, 0, 0}};
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
        EXPECT_THROW(geo.move_to_boundary(), DebugError);
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
    OrangeParams const& geo = this->params();

    EXPECT_EQ(6, geo.num_volumes());
    EXPECT_EQ(12, geo.num_surfaces());
    EXPECT_FALSE(geo.supports_safety());
}

TEST_F(UniversesTest, params)
{
    OrangeParams const& geo = this->params();
    EXPECT_EQ(12, geo.num_volumes());
    EXPECT_EQ(25, geo.num_surfaces());
    EXPECT_EQ(3, geo.max_depth());
    EXPECT_FALSE(geo.supports_safety());

    EXPECT_VEC_SOFT_EQ(Real3({-2, -6, -1}), geo.bbox().lower());
    EXPECT_VEC_SOFT_EQ(Real3({8, 4, 2}), geo.bbox().upper());

    std::vector<std::string> expected = {"[EXTERIOR]",
                                         "inner_a",
                                         "inner_b",
                                         "bobby",
                                         "johnny",
                                         "[EXTERIOR]",
                                         "inner_c",
                                         "a",
                                         "b",
                                         "c",
                                         "[EXTERIOR]",
                                         "patty"};
    std::vector<std::string> actual;
    for (auto const id : range(VolumeId{geo.num_volumes()}))
    {
        actual.push_back(geo.id_to_label(id).name);
    }

    EXPECT_VEC_EQ(expected, actual);
}

TEST_F(UniversesTest, initialize_with_multiple_universes)
{
    auto geo = this->make_track_view();

    // Initialize in outermost universe
    geo = Initializer_t{{-1, -2, 1}, {1, 0, 0}};
    EXPECT_VEC_SOFT_EQ(Real3({-1, -2, 1}), geo.pos());
    EXPECT_VEC_SOFT_EQ(Real3({1, 0, 0}), geo.dir());
    EXPECT_EQ("johnny", this->params().id_to_label(geo.volume_id()).name);
    EXPECT_FALSE(geo.is_outside());
    EXPECT_FALSE(geo.is_on_boundary());

    // Initialize in daughter universe
    geo = Initializer_t{{0.5, -2, 1}, {1, 0, 0}};
    EXPECT_VEC_SOFT_EQ(Real3({0.5, -2, 1}), geo.pos());
    EXPECT_VEC_SOFT_EQ(Real3({1, 0, 0}), geo.dir());
    EXPECT_EQ("c", this->params().id_to_label(geo.volume_id()).name);
    EXPECT_FALSE(geo.is_outside());
    EXPECT_FALSE(geo.is_on_boundary());

    // Initialize in daughter universe using "this == &other"
    geo = OrangeTrackView::DetailedInitializer{geo, {0, 1, 0}};
    EXPECT_VEC_SOFT_EQ(Real3({0.5, -2, 1}), geo.pos());
    EXPECT_VEC_SOFT_EQ(Real3({0, 1, 0}), geo.dir());
    EXPECT_EQ("c", this->params().id_to_label(geo.volume_id()).name);
    EXPECT_FALSE(geo.is_outside());
    EXPECT_FALSE(geo.is_on_boundary());

    {
        // Initialize a separate track slot
        auto other = this->make_track_view(TrackSlotId{1});
        other = OrangeTrackView::DetailedInitializer{geo, {1, 0, 0}};
        EXPECT_VEC_SOFT_EQ(Real3({0.5, -2, 1}), other.pos());
        EXPECT_VEC_SOFT_EQ(Real3({1, 0, 0}), other.dir());
        EXPECT_EQ("c", this->params().id_to_label(other.volume_id()).name);
        EXPECT_FALSE(other.is_outside());
        EXPECT_FALSE(other.is_on_boundary());
    }
}

TEST_F(UniversesTest, move_internal_multiple_universes)
{
    auto geo = this->make_track_view();

    // Initialize in daughter universe
    geo = Initializer_t{{0.5, -2, 1}, {0, 1, 0}};

    // Move internally, then check that the distance to boundary is correct
    geo.move_internal({0.5, -1, 1});
    auto next = geo.find_next_step();
    EXPECT_SOFT_EQ(1, next.distance);

    // Move again, using other move_internal method
    geo.move_internal(0.1);
    next = geo.find_next_step();
    EXPECT_SOFT_EQ(0.9, next.distance);
}

// Set direction in a daughter universe and then make sure the direction is
// correctly returned at the top level
TEST_F(UniversesTest, change_dir_daughter_universe)
{
    auto geo = this->make_track_view();

    // Initialize inside daughter universe a
    geo = Initializer_t{{1.5, -2.0, 1.0}, {1.0, 0.0, 0.0}};

    // Change the direction
    geo.set_dir({0.0, 1.0, 0.0});

    // Get the direction
    EXPECT_VEC_EQ(Real3({0.0, 1.0, 0.0}), geo.dir());
}

// Cross into daughter universe for the case where the hole cell does not share
// a boundary with another with a parent cell
TEST_F(UniversesTest, cross_into_daughter_non_coincident)
{
    auto geo = this->make_track_view();
    geo = Initializer_t{{2, -5, 1}, {0, 1, 0}};

    auto next = geo.find_next_step();
    EXPECT_SOFT_EQ(1, next.distance);

    geo.move_to_boundary();
    EXPECT_EQ("inner_a.my", this->params().id_to_label(geo.surface_id()).name);
    EXPECT_EQ("johnny", this->params().id_to_label(geo.volume_id()).name);
    EXPECT_VEC_SOFT_EQ(Real3({2, -4, 1}), geo.pos());

    // Cross universe boundary
    geo.cross_boundary();
    EXPECT_EQ("inner_a.my", this->params().id_to_label(geo.surface_id()).name);
    EXPECT_EQ("c", this->params().id_to_label(geo.volume_id()).name);
    EXPECT_VEC_SOFT_EQ(Real3({2, -4, 1}), geo.pos());

    // Make sure we can take another step after crossing
    next = geo.find_next_step();
    EXPECT_SOFT_EQ(1, next.distance);

    geo.move_to_boundary();
    EXPECT_EQ("alpha.my", this->params().id_to_label(geo.surface_id()).name);
}

// Cross into parent universe for the case where the hole cell does not share a
// boundary with another with a parent cell
TEST_F(UniversesTest, cross_into_parent_non_coincident)
{
    auto geo = this->make_track_view();
    geo = Initializer_t{{2, -3.5, 1}, {0, -1, 0}};

    auto next = geo.find_next_step();
    EXPECT_SOFT_EQ(0.5, next.distance);
    geo.move_to_boundary();
    EXPECT_EQ("inner_a.my", this->params().id_to_label(geo.surface_id()).name);
    EXPECT_EQ("c", this->params().id_to_label(geo.volume_id()).name);
    EXPECT_VEC_SOFT_EQ(Real3({2, -4, 1}), geo.pos());

    // Cross universe boundary
    geo.cross_boundary();
    EXPECT_EQ("inner_a.my", this->params().id_to_label(geo.surface_id()).name);
    EXPECT_EQ("johnny", this->params().id_to_label(geo.volume_id()).name);
    EXPECT_VEC_SOFT_EQ(Real3({2, -4, 1}), geo.pos());

    // Make sure we can take another step after crossing
    next = geo.find_next_step();
    EXPECT_SOFT_EQ(2, next.distance);

    geo.move_to_boundary();
    EXPECT_EQ("john.my", this->params().id_to_label(geo.surface_id()).name);
}

// Cross into daughter universe for the case where the hole cell shares a
// boundary with another with a parent cell
TEST_F(UniversesTest, cross_into_daughter_coincident)
{
    auto geo = this->make_track_view();
    geo = Initializer_t{{2, 1, 1}, {0, -1, 0}};

    auto next = geo.find_next_step();
    EXPECT_SOFT_EQ(1, next.distance);

    geo.move_to_boundary();
    EXPECT_EQ("bob.my", this->params().id_to_label(geo.surface_id()).name);
    EXPECT_EQ("bobby", this->params().id_to_label(geo.volume_id()).name);
    EXPECT_VEC_SOFT_EQ(Real3({2, 0, 1}), geo.pos());

    // Cross universe boundary
    geo.cross_boundary();
    EXPECT_EQ("bob.my", this->params().id_to_label(geo.surface_id()).name);
    EXPECT_EQ("c", this->params().id_to_label(geo.volume_id()).name);
    EXPECT_VEC_SOFT_EQ(Real3({2, 0, 1}), geo.pos());

    // Make sure we can take another step after crossing
    next = geo.find_next_step();
    EXPECT_SOFT_EQ(1, next.distance);

    geo.move_to_boundary();
    EXPECT_EQ("alpha.py", this->params().id_to_label(geo.surface_id()).name);
}

// Cross into parent universe for the case where the hole cell shares a
// boundary with another with a parent cell
TEST_F(UniversesTest, cross_into_parent_coincident)
{
    auto geo = this->make_track_view();
    geo = Initializer_t{{2, -0.5, 1}, {0, 1, 0}};

    auto next = geo.find_next_step();
    EXPECT_SOFT_EQ(0.5, next.distance);

    geo.move_to_boundary();
    EXPECT_EQ("bob.my", this->params().id_to_label(geo.surface_id()).name);
    EXPECT_EQ("c", this->params().id_to_label(geo.volume_id()).name);
    EXPECT_VEC_SOFT_EQ(Real3({2, 0, 1}), geo.pos());

    // Cross universe boundary
    geo.cross_boundary();
    EXPECT_EQ("bob.my", this->params().id_to_label(geo.surface_id()).name);
    EXPECT_EQ("bobby", this->params().id_to_label(geo.volume_id()).name);
    EXPECT_VEC_SOFT_EQ(Real3({2, 0, 1}), geo.pos());

    // Make sure we can take another step after crossing
    next = geo.find_next_step();
    EXPECT_SOFT_EQ(2, next.distance);

    geo.move_to_boundary();
    EXPECT_EQ("bob.py", this->params().id_to_label(geo.surface_id()).name);
}

// Cross into daughter universe that is two levels down
TEST_F(UniversesTest, cross_into_daughter_doubly_coincident)
{
    auto geo = this->make_track_view();
    geo = Initializer_t{{0.25, -4.5, 1}, {0, 1, 0}};

    auto next = geo.find_next_step();
    EXPECT_SOFT_EQ(0.5, next.distance);

    geo.move_to_boundary();
    EXPECT_EQ("inner_a.my", this->params().id_to_label(geo.surface_id()).name);
    EXPECT_EQ("johnny", this->params().id_to_label(geo.volume_id()).name);
    EXPECT_VEC_SOFT_EQ(Real3({0.25, -4, 1}), geo.pos());

    // Cross universe boundary
    geo.cross_boundary();
    EXPECT_EQ("inner_a.my", this->params().id_to_label(geo.surface_id()).name);
    EXPECT_EQ("patty", this->params().id_to_label(geo.volume_id()).name);
    EXPECT_VEC_SOFT_EQ(Real3({0.25, -4, 1}), geo.pos());

    // Make sure we can take another step after crossing
    next = geo.find_next_step();
    EXPECT_SOFT_EQ(0.5, next.distance);

    geo.move_to_boundary();
    EXPECT_EQ("inner_c.py", this->params().id_to_label(geo.surface_id()).name);
}

// Cross into parent universe that is two levels down
TEST_F(UniversesTest, cross_into_parent_doubly_coincident)
{
    auto geo = this->make_track_view();
    geo = Initializer_t{{0.25, -3.75, 1}, {0, -1, 0}};

    auto next = geo.find_next_step();
    EXPECT_SOFT_EQ(0.25, next.distance);

    geo.move_to_boundary();
    EXPECT_EQ("inner_a.my", this->params().id_to_label(geo.surface_id()).name);
    EXPECT_EQ("patty", this->params().id_to_label(geo.volume_id()).name);
    EXPECT_VEC_SOFT_EQ(Real3({0.25, -4, 1}), geo.pos());

    // Cross universe boundary
    geo.cross_boundary();
    EXPECT_EQ("inner_a.my", this->params().id_to_label(geo.surface_id()).name);
    EXPECT_EQ("johnny", this->params().id_to_label(geo.volume_id()).name);
    EXPECT_VEC_SOFT_EQ(Real3({0.25, -4, 1}), geo.pos());

    // Make sure we can take another step after crossing
    next = geo.find_next_step();
    EXPECT_SOFT_EQ(2, next.distance);

    geo.move_to_boundary();
    EXPECT_EQ("john.my", this->params().id_to_label(geo.surface_id()).name);
}

// Cross between two daughter universes that share a boundary
TEST_F(UniversesTest, cross_between_daughters)
{
    auto geo = this->make_track_view();

    // Initialize in outermost universe
    geo = Initializer_t{{2, -2, 0.7}, {0, 0, -1}};

    auto next = geo.find_next_step();
    EXPECT_SOFT_EQ(0.2, next.distance);

    geo.move_to_boundary();
    EXPECT_EQ("inner_a.pz", this->params().id_to_label(geo.surface_id()).name);
    EXPECT_EQ("a", this->params().id_to_label(geo.volume_id()).name);
    EXPECT_VEC_SOFT_EQ(Real3({2, -2, 0.5}), geo.pos());

    // Cross universe boundary
    geo.cross_boundary();
    EXPECT_EQ("inner_a.pz", this->params().id_to_label(geo.surface_id()).name);
    EXPECT_EQ("a", this->params().id_to_label(geo.volume_id()).name);
    EXPECT_VEC_SOFT_EQ(Real3({2, -2, 0.5}), geo.pos());

    // Make sure we can take another step after crossing
    next = geo.find_next_step();
    EXPECT_SOFT_EQ(1, next.distance);

    geo.move_to_boundary();
    EXPECT_EQ("bob.mz", this->params().id_to_label(geo.surface_id()).name);
}

TEST_F(RectArrayTest, params)
{
    OrangeParams const& geo = this->params();
    EXPECT_EQ(35, geo.num_volumes());
    EXPECT_EQ(22, geo.num_surfaces());
    EXPECT_EQ(4, geo.max_depth());
    EXPECT_FALSE(geo.supports_safety());

    EXPECT_VEC_SOFT_EQ(Real3({-12, -4, -5}), geo.bbox().lower());
    EXPECT_VEC_SOFT_EQ(Real3({12, 10, 5}), geo.bbox().upper());
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

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
