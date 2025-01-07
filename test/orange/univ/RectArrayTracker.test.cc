//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/univ/RectArrayTracker.test.cc
//---------------------------------------------------------------------------//
#include "orange/univ/RectArrayTracker.hh"

#include <algorithm>
#include <random>

#include "corecel/Config.hh"

#include "corecel/data/CollectionAlgorithms.hh"
#include "corecel/data/CollectionStateStore.hh"
#include "corecel/data/Ref.hh"
#include "corecel/io/Repr.hh"
#include "corecel/math/ArrayUtils.hh"
#include "corecel/sys/Stopwatch.hh"
#include "orange/OrangeGeoTestBase.hh"
#include "orange/OrangeParams.hh"
#include "orange/detail/UniverseIndexer.hh"
#include "celeritas/Constants.hh"
#include "celeritas/random/distribution/IsotropicDistribution.hh"
#include "celeritas/random/distribution/UniformBoxDistribution.hh"

#include "SimpleUnitTracker.test.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// TEST FIXTURES
//---------------------------------------------------------------------------//

class RectArrayTrackerTest : public OrangeGeoTestBase
{
  protected:
    using StateHostValue = HostVal<OrangeStateData>;
    using StateHostRef = HostRef<OrangeStateData>;
    using HostStateStore
        = CollectionStateStore<OrangeStateData, MemSpace::host>;
    using Initialization = ::celeritas::detail::Initialization;
    using LocalState = ::celeritas::detail::LocalState;

  protected:
    // Initialization without any logical state
    LocalState make_state(Real3 pos, Real3 dir);

    // Initialization inside a volume
    LocalState make_state(Real3 pos, Real3 dir, LocalVolumeId);

    // Prepare for initialization across a surface
    LocalState make_state_crossing(Real3 pos,
                                   Real3 dir,
                                   LocalVolumeId volid,
                                   LocalSurfaceId surfid,
                                   Sense sense);

    void SetUp() override { this->build_geometry("rect-array.org.json"); }
};

//---------------------------------------------------------------------------//
// TEST FIXTURE IMPLEMENTATION
//---------------------------------------------------------------------------//
/*!
 * Initialize without any logical state.
 */
LocalState RectArrayTrackerTest::make_state(Real3 pos, Real3 dir)
{
    LocalState state;
    state.pos = pos;
    state.dir = make_unit_vector(dir);
    state.volume = {};
    state.surface = {};

    auto const& hsref = this->host_state();
    auto face_storage = hsref.temp_face[AllItems<FaceId>{}];
    state.temp_sense = hsref.temp_sense[AllItems<SenseValue>{}];
    state.temp_next.face = face_storage.data();
    state.temp_next.distance
        = hsref.temp_distance[AllItems<real_type>{}].data();
    state.temp_next.isect = hsref.temp_isect[AllItems<size_type>{}].data();
    state.temp_next.size = face_storage.size();
    return state;
}

//---------------------------------------------------------------------------//
/*!
 * Initialize inside a volume.
 */
LocalState
RectArrayTrackerTest::make_state(Real3 pos, Real3 dir, LocalVolumeId volid)
{
    LocalState state = this->make_state(pos, dir);
    state.volume = volid;
    return state;
}

//---------------------------------------------------------------------------//
/*!
 * Initialize crossing a surface.
 *
 * This takes the *before-crossing volume* and *before-crossing sense*.
 */
LocalState RectArrayTrackerTest::make_state_crossing(Real3 pos,
                                                     Real3 dir,
                                                     LocalVolumeId volid,
                                                     LocalSurfaceId surfid,
                                                     Sense sense)
{
    LocalState state = this->make_state(pos, dir, volid);
    state.surface = {surfid, sense};
    return state;
}

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(RectArrayTrackerTest, initialize)
{
    RectArrayTracker tracker(this->host_params(), RectArrayId{0});
    SCOPED_TRACE("vol {0,0,0}");
    {
        auto init
            = tracker.initialize(this->make_state({0.1, 0.1, 0.1}, {0, 0, 1}));

        this->id_to_label(UniverseId{2}, init.volume);

        EXPECT_EQ("{0,0,0}", this->id_to_label(UniverseId{2}, init.volume));
        EXPECT_FALSE(init.surface);

        EXPECT_EQ(24, tracker.num_volumes());
    }

    SCOPED_TRACE("vol {1,1,0}");
    {
        auto init
            = tracker.initialize(this->make_state({3.1, 3.1, 0.1}, {0, 0, 1}));

        this->id_to_label(UniverseId{2}, init.volume);

        EXPECT_EQ("{1,1,0}", this->id_to_label(UniverseId{2}, init.volume));
        EXPECT_FALSE(init.surface);
    }
}

TEST_F(RectArrayTrackerTest, intersect)
{
    auto inf = std::numeric_limits<real_type>::infinity();

    RectArrayTracker tracker(this->host_params(), RectArrayId{0});

    SCOPED_TRACE(
        "RectArrayTracker, head-on intersections from volume (0, 0, 0)");
    {
        auto isect = tracker.intersect(
            this->make_state({0.1, 0.1, 0.1}, {-1, 0, 0}, LocalVolumeId{0}));
        EXPECT_EQ(inf, isect.distance);
        EXPECT_EQ("[none]",
                  this->id_to_label(UniverseId{2}, isect.surface.id()));
        EXPECT_EQ(Sense::inside, isect.surface.unchecked_sense());

        isect = tracker.intersect(
            this->make_state({0.1, 0.1, 0.1}, {1, 0, 0}, LocalVolumeId{0}));
        EXPECT_SOFT_EQ(2.9, isect.distance);
        EXPECT_EQ("{x,1}",
                  this->id_to_label(UniverseId{2}, isect.surface.id()));
        EXPECT_EQ(Sense::inside, isect.surface.unchecked_sense());

        isect = tracker.intersect(
            this->make_state({0.1, 0.1, 0.1}, {0, -1, 0}, LocalVolumeId{0}));
        EXPECT_EQ(inf, isect.distance);
        EXPECT_EQ("[none]",
                  this->id_to_label(UniverseId{2}, isect.surface.id()));
        EXPECT_EQ(Sense::inside, isect.surface.unchecked_sense());

        isect = tracker.intersect(
            this->make_state({0.1, 0.1, 0.1}, {0, 1, 0}, LocalVolumeId{0}));
        EXPECT_SOFT_EQ(2.9, isect.distance);
        EXPECT_EQ("{y,1}",
                  this->id_to_label(UniverseId{2}, isect.surface.id()));
        EXPECT_EQ(Sense::inside, isect.surface.unchecked_sense());

        isect = tracker.intersect(
            this->make_state({0.1, 0.1, 0.1}, {0, 0, -1}, LocalVolumeId{0}));
        EXPECT_EQ(inf, isect.distance);
        EXPECT_EQ("[none]",
                  this->id_to_label(UniverseId{2}, isect.surface.id()));
        EXPECT_EQ(Sense::inside, isect.surface.unchecked_sense());

        isect = tracker.intersect(
            this->make_state({0.1, 0.1, 0.1}, {0, 0, 1}, LocalVolumeId{0}));
        EXPECT_SOFT_EQ(4.9, isect.distance);
        EXPECT_EQ("{z,1}",
                  this->id_to_label(UniverseId{2}, isect.surface.id()));
        EXPECT_EQ(Sense::inside, isect.surface.unchecked_sense());
    }

    SCOPED_TRACE(
        "RectArrayTracker, head-on intersections from volume (2, 2, 1)");
    {
        auto isect = tracker.intersect(
            this->make_state({10.5, 7.5, 7.5}, {-1, 0, 0}, LocalVolumeId{21}));
        EXPECT_SOFT_EQ(4.5, isect.distance);
        EXPECT_EQ("{x,2}",
                  this->id_to_label(UniverseId{2}, isect.surface.id()));
        EXPECT_EQ(Sense::outside, isect.surface.unchecked_sense());

        isect = tracker.intersect(
            this->make_state({10.5, 7.5, 7.5}, {1, 0, 0}, LocalVolumeId{21}));
        EXPECT_EQ(inf, isect.distance);
        EXPECT_EQ("[none]",
                  this->id_to_label(UniverseId{2}, isect.surface.id()));
        EXPECT_EQ(Sense::inside, isect.surface.unchecked_sense());

        isect = tracker.intersect(
            this->make_state({10.5, 7.5, 7.5}, {0, -1, 0}, LocalVolumeId{21}));
        EXPECT_SOFT_EQ(1.5, isect.distance);
        EXPECT_EQ("{y,2}",
                  this->id_to_label(UniverseId{2}, isect.surface.id()));
        EXPECT_EQ(Sense::outside, isect.surface.unchecked_sense());

        isect = tracker.intersect(
            this->make_state({10.5, 7.5, 7.5}, {0, 1, 0}, LocalVolumeId{21}));
        EXPECT_SOFT_EQ(1.5, isect.distance);
        EXPECT_EQ("{y,3}",
                  this->id_to_label(UniverseId{2}, isect.surface.id()));
        EXPECT_EQ(Sense::inside, isect.surface.unchecked_sense());

        isect = tracker.intersect(
            this->make_state({10.5, 7.5, 7.5}, {0, 0, -1}, LocalVolumeId{21}));
        EXPECT_SOFT_EQ(2.5, isect.distance);
        EXPECT_EQ("{z,1}",
                  this->id_to_label(UniverseId{2}, isect.surface.id()));
        EXPECT_EQ(Sense::outside, isect.surface.unchecked_sense());

        isect = tracker.intersect(
            this->make_state({10.5, 7.5, 7.5}, {0, 0, 1}, LocalVolumeId{21}));
        EXPECT_EQ(inf, isect.distance);
        EXPECT_EQ("[none]",
                  this->id_to_label(UniverseId{2}, isect.surface.id()));
        EXPECT_EQ(Sense::inside, isect.surface.unchecked_sense());
    }

    SCOPED_TRACE("Intersecting at an angle");
    {
        auto isect = tracker.intersect(
            this->make_state({4.5, 4.5, 1}, {1, 1, 0}, LocalVolumeId{15}));
        EXPECT_SOFT_EQ(std::sqrt(2 * 1.5 * 1.5), isect.distance);
        EXPECT_EQ("{x,2}",
                  this->id_to_label(UniverseId{2}, isect.surface.id()));
        EXPECT_EQ(Sense::inside, isect.surface.unchecked_sense());
    }
}

TEST_F(RectArrayTrackerTest, intersect_max_step)
{
    RectArrayTracker tracker(this->host_params(), RectArrayId{0});

    SCOPED_TRACE("Intersecting with max_step parameter specified");
    {
        auto isect = tracker.intersect(
            this->make_state({0.1, 0.1, 0.1}, {0, 0, -1}, LocalVolumeId{0}),
            0.1);
        EXPECT_SOFT_EQ(0.1, isect.distance);
        EXPECT_EQ("[none]",
                  this->id_to_label(UniverseId{2}, isect.surface.id()));
        EXPECT_EQ(Sense::inside, isect.surface.unchecked_sense());

        isect = tracker.intersect(
            this->make_state({0.1, 0.1, 0.1}, {0, 0, -1}, LocalVolumeId{0}),
            0.05);
        EXPECT_SOFT_EQ(0.05, isect.distance);
        EXPECT_EQ("[none]",
                  this->id_to_label(UniverseId{2}, isect.surface.id()));
    }
}

TEST_F(RectArrayTrackerTest, cross_boundary)
{
    RectArrayTracker tracker(this->host_params(), RectArrayId{0});

    SCOPED_TRACE("Crossing between internal cells through x=6 surface");
    {
        auto state = this->make_state_crossing({6, 7, 4},
                                               {-1, 0, 0},
                                               LocalVolumeId{20},
                                               LocalSurfaceId{2},
                                               Sense::inside);
        auto init = tracker.cross_boundary(state);

        EXPECT_EQ(LocalVolumeId{12}, init.volume);
        EXPECT_EQ("{x,2}", this->id_to_label(UniverseId{2}, init.surface.id()));
        EXPECT_EQ(Sense::inside, init.surface.unchecked_sense());
    }

    SCOPED_TRACE("Crossing between internal cells through y=3 surface");
    {
        auto state = this->make_state_crossing({4, 3, 8},
                                               {0, -1, 0},
                                               LocalVolumeId{11},
                                               LocalSurfaceId{5},
                                               Sense::inside);
        auto init = tracker.cross_boundary(state);

        EXPECT_EQ(LocalVolumeId{9}, init.volume);
        EXPECT_EQ("{y,1}", this->id_to_label(UniverseId{2}, init.surface.id()));
        EXPECT_EQ(Sense::inside, init.surface.unchecked_sense());
    }

    SCOPED_TRACE("Crossing between internal cells through z=5 surface");
    {
        auto state = this->make_state_crossing({4.5, 6.5, 5},
                                               {0, 0, 1},
                                               LocalVolumeId{12},
                                               LocalSurfaceId{10},
                                               Sense::outside);
        auto init = tracker.cross_boundary(state);

        EXPECT_EQ(LocalVolumeId{13}, init.volume);
        EXPECT_EQ("{z,1}", this->id_to_label(UniverseId{2}, init.surface.id()));
        EXPECT_EQ(Sense::outside, init.surface.unchecked_sense());
    }
}

TEST_F(RectArrayTrackerTest, safety)
{
    RectArrayTracker tracker(this->host_params(), RectArrayId{0});

    SCOPED_TRACE("In volume {0, 0, 0}");
    {
        auto volid = LocalVolumeId{0};

        EXPECT_SOFT_EQ(2.5, tracker.safety({0.11, 0.5, 0.5}, volid));
        EXPECT_SOFT_EQ(0.08, tracker.safety({2.92, 0.5, 0.5}, volid));
        EXPECT_SOFT_EQ(2.5, tracker.safety({0.5, 0.11, 0.5}, volid));
        EXPECT_SOFT_EQ(0.08, tracker.safety({0.5, 2.92, 0.5}, volid));
        EXPECT_SOFT_EQ(2.5, tracker.safety({0.5, 0.11, 0.01}, volid));
        EXPECT_SOFT_EQ(0.02, tracker.safety({0.5, 2.92, 4.98}, volid));
    }

    SCOPED_TRACE("In volume {2, 2, 1}");
    {
        auto volid = LocalVolumeId{21};

        EXPECT_SOFT_EQ(0.04, tracker.safety({6.04, 6.5, 5.5}, volid));
        EXPECT_SOFT_EQ(0.02, tracker.safety({11.98, 6.5, 5.02}, volid));
        EXPECT_SOFT_EQ(0.04, tracker.safety({6.5, 6.04, 5.5}, volid));
        EXPECT_SOFT_EQ(0.02, tracker.safety({6.5, 8.98, 5.5}, volid));
        EXPECT_SOFT_EQ(0.01, tracker.safety({6.5, 6.04, 5.01}, volid));
        EXPECT_SOFT_EQ(0.01, tracker.safety({6.01, 8.98, 9.99}, volid));
    }
}

TEST_F(RectArrayTrackerTest, normal)
{
    RectArrayTracker tracker(this->host_params(), RectArrayId{0});

    SCOPED_TRACE("X normals");
    {
        EXPECT_VEC_SOFT_EQ(Real3({
                               1.,
                               0.,
                               0.,
                           }),
                           tracker.normal({0.0, 0.5, 0.5}, LocalSurfaceId{0}));
        EXPECT_VEC_SOFT_EQ(Real3({
                               1.,
                               0.,
                               0.,
                           }),
                           tracker.normal({6.0, 0.5, 0.5}, LocalSurfaceId{2}));
        EXPECT_VEC_SOFT_EQ(Real3({
                               1.,
                               0.,
                               0.,
                           }),
                           tracker.normal({12, 0.5, 0.5}, LocalSurfaceId{3}));
    }

    SCOPED_TRACE("Y normals");
    {
        EXPECT_VEC_SOFT_EQ(Real3({
                               0.,
                               1.,
                               0.,
                           }),
                           tracker.normal({1.5, 0., 0.5}, LocalSurfaceId{4}));
        EXPECT_VEC_SOFT_EQ(Real3({
                               0.,
                               1.,
                               0.,
                           }),
                           tracker.normal({1.5, 3., 0.5}, LocalSurfaceId{5}));
        EXPECT_VEC_SOFT_EQ(Real3({
                               0.,
                               1.,
                               0.,
                           }),
                           tracker.normal({1.5, 12., 0.5}, LocalSurfaceId{8}));
    }

    SCOPED_TRACE("Z normals");
    {
        EXPECT_VEC_SOFT_EQ(Real3({
                               0.,
                               0.,
                               1.,
                           }),
                           tracker.normal({4., 5., 0.}, LocalSurfaceId{9}));
        EXPECT_VEC_SOFT_EQ(Real3({
                               0.,
                               0.,
                               1.,
                           }),
                           tracker.normal({4., 5., 5.}, LocalSurfaceId{10}));
        EXPECT_VEC_SOFT_EQ(Real3({
                               0.,
                               0.,
                               1.,
                           }),
                           tracker.normal({4., 5., 10.}, LocalSurfaceId{11}));
    }
}

TEST_F(RectArrayTrackerTest, daughter)
{
    RectArrayTracker tracker(this->host_params(), RectArrayId{0});

    EXPECT_EQ(2, tracker.daughter(LocalVolumeId{0}).get());
    EXPECT_EQ(3, tracker.daughter(LocalVolumeId{1}).get());
    EXPECT_EQ(4, tracker.daughter(LocalVolumeId{2}).get());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
