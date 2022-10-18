//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/univ/SimpleUnitTracker.test.cc
//---------------------------------------------------------------------------//
#include "orange/univ/SimpleUnitTracker.hh"

#include <algorithm>
#include <random>

#include "celeritas_config.h"
#include "corecel/data/CollectionAlgorithms.hh"
#include "corecel/data/CollectionStateStore.hh"
#include "corecel/io/Repr.hh"
#include "corecel/math/ArrayUtils.hh"
#include "corecel/sys/Stopwatch.hh"
#include "orange/OrangeGeoTestBase.hh"
#include "celeritas/Constants.hh"
#include "celeritas/random/distribution/IsotropicDistribution.hh"
#include "celeritas/random/distribution/UniformBoxDistribution.hh"

#include "SimpleUnitTracker.test.hh"
#include "celeritas_test.hh"

using celeritas::constants::sqrt_three;
using celeritas::constants::sqrt_two;

namespace celeritas
{
namespace test
{
namespace
{
constexpr real_type sqrt_half = sqrt_two / 2;
}

//---------------------------------------------------------------------------//
// TEST FIXTURES
//---------------------------------------------------------------------------//

class SimpleUnitTrackerTest : public OrangeGeoTestBase
{
  protected:
    using StateHostValue = HostVal<OrangeStateData>;
    using StateHostRef   = HostRef<OrangeStateData>;
    using HostStateStore
        = CollectionStateStore<OrangeStateData, MemSpace::host>;
    using Initialization = ::celeritas::detail::Initialization;
    using LocalState     = ::celeritas::detail::LocalState;

    struct HeuristicInitResult
    {
        std::vector<double> vol_fractions; //!< Fraction per volume ID
        double              failed{0}; //!< Fraction that couldn't initialize
        double              walltime_per_track_ns{0}; //!< Kernel time

        void print_expected() const;
    };

  protected:
    // Initialization without any logical state
    LocalState make_state(Real3 pos, Real3 dir);

    // Initialization inside a volume
    LocalState make_state(Real3 pos, Real3 dir, const char* vol);

    // Initialization on a surface
    LocalState make_state(
        Real3 pos, Real3 dir, const char* vol, const char* surf, char sense);

    // Prepare for initialization across a surface
    LocalState make_state_crossing(
        Real3 pos, Real3 dir, const char* vol, const char* surf, char sense);

    HeuristicInitResult run_heuristic_init_host(size_type num_tracks) const;
    HeuristicInitResult run_heuristic_init_device(size_type num_tracks) const;

  private:
    StateHostValue setup_heuristic_states(size_type num_tracks) const;
    HeuristicInitResult
    reduce_heuristic_init(const StateHostRef&, double) const;
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

#define FieldLayersTest TEST_IF_CELERITAS_JSON(FieldLayersTest)
class FieldLayersTest : public SimpleUnitTrackerTest
{
    void SetUp() override { this->build_geometry("field-layers.org.json"); }
};

#define FiveVolumesTest TEST_IF_CELERITAS_JSON(FiveVolumesTest)
class FiveVolumesTest : public SimpleUnitTrackerTest
{
    void SetUp() override { this->build_geometry("five-volumes.org.json"); }
};

//---------------------------------------------------------------------------//
// TEST FIXTURE IMPLEMENTATION
//---------------------------------------------------------------------------//
/*!
 * Initialize without any logical state.
 */
LocalState SimpleUnitTrackerTest::make_state(Real3 pos, Real3 dir)
{
    normalize_direction(&dir);
    LocalState state;
    state.pos     = pos;
    state.dir     = dir;
    state.volume  = {};
    state.surface = {};

    const auto& hsref        = this->host_state();
    auto        face_storage = hsref.temp_face[AllItems<FaceId>{}];
    state.temp_sense         = hsref.temp_sense[AllItems<Sense>{}];
    state.temp_next.face     = face_storage.data();
    state.temp_next.distance
        = hsref.temp_distance[AllItems<real_type>{}].data();
    state.temp_next.isect = hsref.temp_isect[AllItems<size_type>{}].data();
    state.temp_next.size  = face_storage.size();
    return state;
}

//---------------------------------------------------------------------------//
/*!
 * Initialize inside a volume.
 */
LocalState
SimpleUnitTrackerTest::make_state(Real3 pos, Real3 dir, const char* vol)
{
    LocalState state = this->make_state(pos, dir);
    state.volume     = this->find_volume(vol);
    return state;
}

//---------------------------------------------------------------------------//
/*!
 * Initialize crossing a surface.
 *
 * This takes the *before-crossing volume* and *before-crossing sense*.
 */
LocalState SimpleUnitTrackerTest::make_state(
    Real3 pos, Real3 dir, const char* vol, const char* surf, char sense)
{
    CELER_ASSERT(vol && surf);
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
            CELER_VALIDATE(false, << "invalid sense value '" << sense << "'");
    }

    LocalState state = this->make_state(pos, dir);
    state.volume     = this->find_volume(vol);
    // *Intentionally* flip the sense because we're looking for the
    // post-crossing volume. This is normally done by the multi-level
    // TrackingGeometry.
    state.surface = {this->find_surface(surf), before_crossing_sense};
    return state;
}

//---------------------------------------------------------------------------//
/*!
 * Initialize crossing a surface.
 *
 * This takes the *before-crossing volume* and *before-crossing sense*.
 */
LocalState SimpleUnitTrackerTest::make_state_crossing(
    Real3 pos, Real3 dir, const char* vol, const char* surf, char sense)
{
    auto state = this->make_state(pos, dir, vol, surf, sense);
    state.surface
        = {state.surface.id(), flip_sense(state.surface.unchecked_sense())};
    return state;
}

//---------------------------------------------------------------------------//
/*!
 * Initialize particles randomly and tally their resulting locations.
 *
 * This is (in effect) point sampling the bounding box to determine volumes.
 */
auto SimpleUnitTrackerTest::run_heuristic_init_host(size_type num_tracks) const
    -> HeuristicInitResult
{
    HostStateStore states(this->setup_heuristic_states(num_tracks));

    // Set up for host run
    InitializingLauncher<> calc_init{this->params().host_ref(), states.ref()};

    // Loop over all threads
    Stopwatch get_time;
    for (auto tid : range(ThreadId{states.size()}))
    {
        calc_init(tid);
    }

    return this->reduce_heuristic_init(states.ref(), get_time());
}

//---------------------------------------------------------------------------//
/*!
 * Initialize particles randomly and tally their resulting locations.
 */
auto SimpleUnitTrackerTest::run_heuristic_init_device(size_type num_tracks) const
    -> HeuristicInitResult
{
    using DStateStore = CollectionStateStore<OrangeStateData, MemSpace::device>;
    DStateStore states(this->setup_heuristic_states(num_tracks));

    // Run on device
    Stopwatch get_time;
    test_initialize(this->params().device_ref(), states.ref());
    const double kernel_time = get_time();

    // Copy result back to host
    HostStateStore state_host(std::move(states));
    return this->reduce_heuristic_init(state_host.ref(), kernel_time);
}

//---------------------------------------------------------------------------//
/*!
 * Construct states on the host.
 */
auto SimpleUnitTrackerTest::setup_heuristic_states(size_type num_tracks) const
    -> StateHostValue
{
    CELER_EXPECT(num_tracks > 0);
    StateHostValue result;
    resize(&result, this->params().host_ref(), num_tracks);
    auto pos_view = result.pos[AllItems<Real3>{}];
    auto dir_view = result.dir[AllItems<Real3>{}];

    std::mt19937 rng;

    // Sample uniform in space and isotropic in direction
    const auto&              bbox = this->params().bbox();
    UniformBoxDistribution<> sample_box{bbox.lower(), bbox.upper()};
    IsotropicDistribution<>  sample_isotropic;
    for (auto i : range(num_tracks))
    {
        pos_view[i] = sample_box(rng);
        dir_view[i] = sample_isotropic(rng);
    }

    // Clear other data
    fill(VolumeId{}, &result.vol);
    fill(SurfaceId{}, &result.surf);

    CELER_ENSURE(result);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Process "heuristic init" test results.
 */
auto SimpleUnitTrackerTest::reduce_heuristic_init(const StateHostRef& host,
                                                  double wall_time) const
    -> HeuristicInitResult
{
    CELER_EXPECT(host);
    CELER_EXPECT(wall_time > 0);
    std::vector<size_type> counts(this->num_volumes());
    size_type              error_count{};

    for (auto vol : host.vol[AllItems<VolumeId>{}])
    {
        if (vol < counts.size())
        {
            ++counts[vol.unchecked_get()];
        }
        else
        {
            ++error_count;
        }
    }

    HeuristicInitResult result;
    result.vol_fractions.resize(counts.size());
    const double norm = 1.0 / static_cast<double>(host.size());
    for (auto i : range(counts.size()))
    {
        result.vol_fractions[i] = norm * static_cast<double>(counts[i]);
    }
    result.failed                = norm * error_count;
    result.walltime_per_track_ns = norm * wall_time * 1e9;
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Output copy-pasteable "gold" comparison unit testing code.
 */
void SimpleUnitTrackerTest::HeuristicInitResult::print_expected() const
{
    cout << "/*** ADD THE FOLLOWING UNIT TEST CODE ***/\n"
         << "static const double expected_vol_fractions[] = "
         << repr(this->vol_fractions) << ";\n"
         << "EXPECT_VEC_SOFT_EQ(expected_vol_fractions, "
            "result.vol_fractions);\n"
         << "EXPECT_SOFT_EQ(" << this->failed << ", result.failed);\n"
         << "// Wall time (ns): " << this->walltime_per_track_ns << "\n"
         << "/*** END CODE ***/\n";
}

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(OneVolumeTest, initialize)
{
    SimpleUnitTracker tracker(this->params().host_ref(), SimpleUnitId{0});

    {
        // Anywhere is valid
        auto init = tracker.initialize(this->make_state({1, 2, 3}, {0, 0, 1}));
        EXPECT_TRUE(init);
        EXPECT_EQ(VolumeId{0}, init.volume);
        EXPECT_FALSE(init.surface);
    }
}

TEST_F(OneVolumeTest, intersect)
{
    SimpleUnitTracker tracker(this->params().host_ref(), SimpleUnitId{0});

    {
        auto state = this->make_state({1, 2, 3}, {0, 0, 1}, "infinite");
        auto isect = tracker.intersect(state);
        EXPECT_FALSE(isect);
        EXPECT_EQ(no_intersection(), isect.distance);

        isect = tracker.intersect(state, 5.0);
        EXPECT_FALSE(isect);
        EXPECT_EQ(5.0, isect.distance);
    }
}

TEST_F(OneVolumeTest, safety)
{
    SimpleUnitTracker tracker(this->params().host_ref(), SimpleUnitId{0});

    EXPECT_SOFT_EQ(inf,
                   tracker.safety({1, 2, 3}, this->find_volume("infinite")));
}

TEST_F(OneVolumeTest, heuristic_init)
{
    size_type num_tracks = 1024;

    static const double expected_vol_fractions[] = {1.0};

    {
        SCOPED_TRACE("Host heuristic");
        auto result = this->run_heuristic_init_host(num_tracks);

        EXPECT_VEC_SOFT_EQ(expected_vol_fractions, result.vol_fractions);
        EXPECT_SOFT_EQ(0, result.failed);
    }
    if (CELER_USE_DEVICE)
    {
        SCOPED_TRACE("Device heuristic");
        auto result = this->run_heuristic_init_device(num_tracks);
        EXPECT_VEC_SOFT_EQ(expected_vol_fractions, result.vol_fractions);
        EXPECT_SOFT_EQ(0, result.failed);
    }
}

//---------------------------------------------------------------------------//

TEST_F(TwoVolumeTest, initialize)
{
    SimpleUnitTracker tracker(this->params().host_ref(), SimpleUnitId{0});

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
        SCOPED_TRACE("Outside the sphere");
        auto init
            = tracker.initialize(this->make_state({3.0, 0, 0}, {0, 0, 1}));
        EXPECT_EQ("outside", this->id_to_label(init.volume));
        EXPECT_FALSE(init.surface);
    }
}

TEST_F(TwoVolumeTest, cross_boundary)
{
    SimpleUnitTracker tracker(this->params().host_ref(), SimpleUnitId{0});

    {
        SCOPED_TRACE("Crossing the boundary from the inside");
        auto init = tracker.cross_boundary(this->make_state_crossing(
            {1.5, 0, 0}, {0, 0, 1}, "inside", "sphere", '-'));
        EXPECT_EQ("outside", this->id_to_label(init.volume));
        EXPECT_EQ("sphere", this->id_to_label(init.surface.id()));
        EXPECT_EQ(Sense::outside, init.surface.sense());
    }
    {
        SCOPED_TRACE("Crossing the boundary from the outside");
        auto init = tracker.cross_boundary(this->make_state_crossing(
            {1.5, 0, 0}, {0, 0, 1}, "outside", "sphere", '+'));
        EXPECT_EQ("inside", this->id_to_label(init.volume));
        EXPECT_EQ("sphere", this->id_to_label(init.surface.id()));
        EXPECT_EQ(Sense::inside, init.surface.sense());
    }
}

TEST_F(TwoVolumeTest, intersect)
{
    SimpleUnitTracker tracker(this->params().host_ref(), SimpleUnitId{0});

    {
        SCOPED_TRACE("Inside");
        auto state = this->make_state({0.5, 0, 0}, {0, 0, 1}, "inside");
        auto isect = tracker.intersect(state);
        EXPECT_TRUE(isect);
        EXPECT_EQ(SurfaceId{0}, isect.surface.id());
        EXPECT_EQ(Sense::inside, isect.surface.unchecked_sense());
        EXPECT_SOFT_EQ(sqrt_two, isect.distance);

        state = this->make_state({0.5, 0, 0}, {1, 0, 0}, "inside");
        isect = tracker.intersect(state);
        EXPECT_TRUE(isect);
        EXPECT_EQ(SurfaceId{0}, isect.surface.id());
        EXPECT_EQ(Sense::inside, isect.surface.unchecked_sense());
        EXPECT_SOFT_EQ(1.0, isect.distance);

        // Range limit: further than surface
        isect = tracker.intersect(state, 10.0);
        EXPECT_TRUE(isect);
        EXPECT_EQ(SurfaceId{0}, isect.surface.id());
        EXPECT_EQ(Sense::inside, isect.surface.unchecked_sense());
        EXPECT_SOFT_EQ(1.0, isect.distance);

        // Coincident
        isect = tracker.intersect(state, 1.0);
        EXPECT_TRUE(isect);
        EXPECT_EQ(SurfaceId{0}, isect.surface.id());
        EXPECT_EQ(Sense::inside, isect.surface.unchecked_sense());
        EXPECT_SOFT_EQ(1.0, isect.distance);

        // Range limit: less than
        isect = tracker.intersect(state, 0.9);
        EXPECT_FALSE(isect);
        EXPECT_SOFT_EQ(0.9, isect.distance);
    }
    {
        SCOPED_TRACE("Outside");
        auto state = this->make_state({0, 0, 2.0}, {0, 0, 1}, "outside");
        auto isect = tracker.intersect(state);
        EXPECT_FALSE(isect);

        state = this->make_state({0, 0, 2.0}, {0, 0, -1}, "inside");
        isect = tracker.intersect(state);
        EXPECT_TRUE(isect);
        EXPECT_EQ(SurfaceId{0}, isect.surface.id());
        EXPECT_EQ(Sense::outside, isect.surface.unchecked_sense());
        EXPECT_SOFT_EQ(0.5, isect.distance);
    }
    {
        SCOPED_TRACE("Outside on surface heading out");
        auto state = this->make_state(
            {0, 0, 1.5}, {0, 0, 1}, "outside", "sphere", '+');
        auto isect = tracker.intersect(state);
        EXPECT_FALSE(isect);
    }
    {
        SCOPED_TRACE("Outside on surface heading in");
        auto state = this->make_state(
            {0, 0, 1.5}, {0, 0, -1}, "outside", "sphere", '+');
        auto isect = tracker.intersect(state);
        EXPECT_EQ(SurfaceId{0}, isect.surface.id());
        EXPECT_EQ(Sense::outside, isect.surface.unchecked_sense());
        // NOTE: being on a surface with opposite sense and direction means
        // we should get a "zero distance" movement plus sense change, but
        // zero-distance movements are prohibited. This will only happen in
        // practice by changing direction immediately after crossing a surface
        // without moving (impossible) since the tracking geometry does not
        // allow initialization on a surface.
#if 0
        // "Correct" result
        EXPECT_SOFT_EQ(0.0, isect.distance);
#else
        // "Expected" result
        EXPECT_SOFT_EQ(3.0, isect.distance);
#endif
    }
    {
        SCOPED_TRACE("Inside on surface heading in");
        auto state = this->make_state(
            {0, 0, 1.5}, {0, 0, -1}, "inside", "sphere", '-');
        auto isect = tracker.intersect(state);
        EXPECT_EQ(SurfaceId{0}, isect.surface.id());
        EXPECT_EQ(Sense::inside, isect.surface.unchecked_sense());
        EXPECT_SOFT_EQ(3.0, isect.distance);
    }
    {
        SCOPED_TRACE("Inside on surface heading out");
        auto state = this->make_state(
            {0, 1.5, 0}, {0, 1, 0}, "inside", "sphere", '-');
        auto isect = tracker.intersect(state);
#if 0
        // "Correct" result when accounting for sense in distance-to-boundary
        // calculation
        EXPECT_EQ(SurfaceId{0}, isect.surface.id());
        EXPECT_EQ(Sense::inside, isect.surface.unchecked_sense());
        EXPECT_SOFT_EQ(0.0, isect.distance);
#else
        // "Expected" result: no intersection
        EXPECT_FALSE(isect);
#endif
    }
}

TEST_F(TwoVolumeTest, safety)
{
    SimpleUnitTracker tracker(this->params().host_ref(), SimpleUnitId{0});
    VolumeId          outside = this->find_volume("outside");
    VolumeId          inside  = this->find_volume("inside");

    EXPECT_SOFT_EQ(1.9641016151377535, tracker.safety({2, 2, 2}, outside));
    EXPECT_SOFT_EQ(1.3284271247461905, tracker.safety({2, 0, 2}, outside));
    EXPECT_SOFT_EQ(0.5, tracker.safety({0, 0, 2}, outside));
    EXPECT_SOFT_EQ(0.5, tracker.safety({0, 0, 1}, inside));
    EXPECT_SOFT_EQ(1.5 - 1e-10, tracker.safety({1e-10, 0, 0}, inside));
#if 0
    // Correct result but there is a singularity at zero
    EXPECT_SOFT_EQ(1.5, tracker.safety({0, 0, 0}, inside)); // degenerate!
#else
    // Actual result
    EXPECT_SOFT_EQ(inf, tracker.safety({0, 0, 0}, inside)); // degenerate!
#endif
}

TEST_F(TwoVolumeTest, normal)
{
    SimpleUnitTracker tracker(this->params().host_ref(), SimpleUnitId{0});

    if (CELERITAS_DEBUG)
    {
        SCOPED_TRACE("Not on a surface");
        EXPECT_THROW(tracker.normal(Real3{0, 0, 1.6}, SurfaceId{}), DebugError);
    }
    {
        Real3 pos{3, -2, 1};
        Real3 expected_normal;
        auto  invnorm = 1 / norm(pos);
        for (auto i : range(3))
        {
            expected_normal[i] = pos[i] * invnorm;
            pos[i]             = expected_normal[i] * real_type(1.5); // radius
        }

        auto actual_normal = tracker.normal(pos, SurfaceId{0});
        EXPECT_VEC_SOFT_EQ(expected_normal, actual_normal);
    }
}

TEST_F(TwoVolumeTest, heuristic_init)
{
    size_type num_tracks = 1024;

    static const double expected_vol_fractions[] = {0.4765625, 0.5234375};

    {
        SCOPED_TRACE("Host heuristic");
        auto result = this->run_heuristic_init_host(num_tracks);

        EXPECT_VEC_SOFT_EQ(expected_vol_fractions, result.vol_fractions);
        EXPECT_SOFT_EQ(0, result.failed);
    }
    if (CELER_USE_DEVICE)
    {
        SCOPED_TRACE("Device heuristic");
        auto result = this->run_heuristic_init_device(num_tracks);
        EXPECT_VEC_SOFT_EQ(expected_vol_fractions, result.vol_fractions);
        EXPECT_SOFT_EQ(0, result.failed);
    }
}

//---------------------------------------------------------------------------//

TEST_F(FieldLayersTest, initialize)
{
    SimpleUnitTracker tracker(this->params().host_ref(), SimpleUnitId{0});

    {
        SCOPED_TRACE("Exterior");
        auto init
            = tracker.initialize(this->make_state({0, 50, 0}, {0, -1, 0}));
        EXPECT_EQ("[EXTERIOR]", this->id_to_label(init.volume));
        EXPECT_FALSE(init.surface);
    }
    {
        auto init = tracker.initialize(this->make_state({0, -3, 0}, {0, 0, 1}));
        EXPECT_EQ("world", this->id_to_label(init.volume));
        EXPECT_FALSE(init.surface);
    }
    {
        auto init
            = tracker.initialize(this->make_state({0, -2.4, 0}, {0, 0, 1}));
        EXPECT_EQ("layer1", this->id_to_label(init.volume));
        EXPECT_FALSE(init.surface);
    }
}

TEST_F(FieldLayersTest, cross_boundary)
{
    SimpleUnitTracker tracker(this->params().host_ref(), SimpleUnitId{0});

    // Test crossing into and out of the "background" with varying levels
    // of numerical imprecision
    for (real_type eps : {-1e-6, -1e-10, -1e-14, 0.0, 1e-14, 1e-10, 1e-6})
    {
        SCOPED_TRACE(eps);
        {
            // From background to volume
            auto init = tracker.cross_boundary(this->make_state_crossing(
                {0, -1.5 + eps, 0}, {0, -1, 0}, "world", "layerbox1.py", '+'));
            EXPECT_EQ("layer1", this->id_to_label(init.volume));
            EXPECT_EQ("layerbox1.py", this->id_to_label(init.surface.id()));
            EXPECT_EQ(Sense::inside, init.surface.unchecked_sense());
        }
        {
            // From volume to background
            auto init = tracker.cross_boundary(this->make_state_crossing(
                {0, -2.5 - eps, 0}, {0, -1, 0}, "layer1", "layerbox1.my", '+'));
            EXPECT_EQ("world", this->id_to_label(init.volume));
            EXPECT_EQ("layerbox1.my", this->id_to_label(init.surface.id()));
            EXPECT_EQ(Sense::inside, init.surface.unchecked_sense());
        }
    }
}

TEST_F(FieldLayersTest, intersect)
{
    SimpleUnitTracker tracker(this->params().host_ref(), SimpleUnitId{0});

    {
        SCOPED_TRACE("straightforward");
        auto state = this->make_state({0, -1, 0}, {0, 1, 0}, "world");
        auto isect = tracker.intersect(state);
        EXPECT_TRUE(isect);
        EXPECT_EQ("layerbox2.my", this->id_to_label(isect.surface.id()));
        EXPECT_EQ(Sense::inside, isect.surface.unchecked_sense());
        EXPECT_SOFT_EQ(0.5, isect.distance);
    }
    {
        SCOPED_TRACE("crossing internal planes");
        auto state = this->make_state({9.6, 4, 9.7}, {-1, -1, -1}, "world");
        auto isect = tracker.intersect(state);
        EXPECT_TRUE(isect);
        EXPECT_EQ("layerbox3.py", this->id_to_label(isect.surface.id()));
        EXPECT_EQ(Sense::outside, isect.surface.unchecked_sense());
        EXPECT_SOFT_EQ(1.5 * sqrt_three, isect.distance);
    }
}

TEST_F(FieldLayersTest, heuristic_init)
{
    size_type           num_tracks               = 8192;
    static const double expected_vol_fractions[] = {0,
                                                    0.018310546875,
                                                    0.019775390625,
                                                    0.020263671875,
                                                    0.0189208984375,
                                                    0.021484375,
                                                    0.9012451171875};

    {
        SCOPED_TRACE("Host heuristic");
        auto result = this->run_heuristic_init_host(num_tracks);
        EXPECT_VEC_SOFT_EQ(expected_vol_fractions, result.vol_fractions);
        EXPECT_SOFT_EQ(0, result.failed);
    }

    if (CELER_USE_DEVICE)
    {
        SCOPED_TRACE("Device heuristic");
        auto result = this->run_heuristic_init_device(num_tracks);
        EXPECT_VEC_SOFT_EQ(expected_vol_fractions, result.vol_fractions);
        EXPECT_SOFT_EQ(0, result.failed);
    }
}

//---------------------------------------------------------------------------//

TEST_F(FiveVolumesTest, properties)
{
    // NOTE: bbox in the JSON file has been adjusted manually.
    const auto& bbox = this->params().bbox();
    EXPECT_VEC_SOFT_EQ(Real3({-1.5, -1.5, -0.5}), bbox.lower());
    EXPECT_VEC_SOFT_EQ(Real3({1.5, 1.5, 0.5}), bbox.upper());
}

TEST_F(FiveVolumesTest, initialize)
{
    SimpleUnitTracker tracker(this->params().host_ref(), SimpleUnitId{0});

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
}

TEST_F(FiveVolumesTest, cross_boundary)
{
    SimpleUnitTracker tracker(this->params().host_ref(), SimpleUnitId{0});

    {
        SCOPED_TRACE("Crossing the boundary from the inside of 'e'");
        auto init = tracker.cross_boundary(this->make_state_crossing(
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
        auto      init = tracker.cross_boundary(this->make_state_crossing(
            {eps, -0.25, 0}, {1, 0, 0}, "e", "epsilon.s", '-'));
        EXPECT_EQ("c", this->id_to_label(init.volume));
        EXPECT_EQ("epsilon.s", this->id_to_label(init.surface.id()));
        EXPECT_EQ(Sense::outside, init.surface.sense());
    }
    {
        SCOPED_TRACE("Crossing the boundary into a more complicated region");
        auto init = tracker.cross_boundary(
            this->make_state_crossing({-.75 * sqrt_half, 0.75 * sqrt_half, 0},
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
        auto init = tracker.cross_boundary(
            this->make_state_crossing({-.75 * sqrt_half, 0.75 * sqrt_half, 0},
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
        // boundary.
        SCOPED_TRACE("Crossing at triple point");
        auto init = tracker.cross_boundary(this->make_state_crossing(
            {0, 0.75, 0}, {0, 1, 0}, "c", "gamma.s", '-'));
        EXPECT_EQ("d", this->id_to_label(init.volume));
        EXPECT_EQ("gamma.s", this->id_to_label(init.surface.id()));
        EXPECT_EQ(Sense::outside, init.surface.sense());

        init = tracker.cross_boundary(this->make_state_crossing(
            {0, 0.75, 0}, {-1, 0, 0}, "d", "gamma.s", '+'));
        EXPECT_EQ("c", this->id_to_label(init.volume));
        EXPECT_EQ("gamma.s", this->id_to_label(init.surface.id()));
        EXPECT_EQ(Sense::inside, init.surface.sense());

        // Near triple point, on sphere but crossing plane edge
        init = tracker.cross_boundary(this->make_state_crossing(
            {0, 0.75, 0}, {-1, 0, 0}, "d", "alpha.px", '+'));
        EXPECT_EQ("a", this->id_to_label(init.volume));
        EXPECT_EQ("alpha.px", this->id_to_label(init.surface.id()));
        EXPECT_EQ(Sense::inside, init.surface.sense());
    }
}

TEST_F(FiveVolumesTest, intersect)
{
    SimpleUnitTracker tracker(this->params().host_ref(), SimpleUnitId{0});

    {
        SCOPED_TRACE("internal surface for a");
        auto state = this->make_state({-0.75, 0.5, 0}, {1, 0, 0}, "a");
        auto isect = tracker.intersect(state);
        EXPECT_TRUE(isect);
        EXPECT_EQ("gamma.s", this->id_to_label(isect.surface.id()));
        EXPECT_EQ(Sense::outside, isect.surface.unchecked_sense());
        EXPECT_SOFT_EQ(0.19098300562505255, isect.distance);
    }
    {
        SCOPED_TRACE("skip internal surfaces for d");
        auto state = this->make_state({-2, 0.5, 0}, {1, 1, 0}, "d");
        auto isect = tracker.intersect(state);
        EXPECT_TRUE(isect);
        EXPECT_EQ("outer.s", this->id_to_label(isect.surface.id()));
        EXPECT_EQ(Sense::inside, isect.surface.unchecked_sense());
        EXPECT_SOFT_EQ(101.04503395088592, isect.distance);

        isect = tracker.intersect(state, 105.0);
        EXPECT_TRUE(isect);
        EXPECT_EQ("outer.s", this->id_to_label(isect.surface.id()));
        EXPECT_EQ(Sense::inside, isect.surface.unchecked_sense());
        EXPECT_SOFT_EQ(101.04503395088592, isect.distance);

        isect = tracker.intersect(state, 100.0);
        EXPECT_FALSE(isect);
        EXPECT_SOFT_EQ(100.0, isect.distance);

        isect = tracker.intersect(state, 1e-12);
        EXPECT_FALSE(isect);
        EXPECT_SOFT_EQ(1e-12, isect.distance);
    }
}

TEST_F(FiveVolumesTest, safety)
{
    SimpleUnitTracker tracker(this->params().host_ref(), SimpleUnitId{0});

    VolumeId a = this->find_volume("a");
    VolumeId d = this->find_volume("d");

    EXPECT_SOFT_EQ(0.15138781886599728, tracker.safety({-0.75, 0.5, 0}, a));

    // Note: distance with "d" surfaces is conservatively to internal planes
    EXPECT_SOFT_EQ(0.5, tracker.safety({-5, 20, 0}, d));
}

TEST_F(FiveVolumesTest, heuristic_init)
{
    size_type num_tracks = 10000;

    static const double expected_vol_fractions[]
        = {0, 0.0701, 0.106, 0.1621, 0.6555, 0.0063};

    {
        SCOPED_TRACE("Host heuristic");
        auto result = this->run_heuristic_init_host(num_tracks);
        EXPECT_VEC_SOFT_EQ(expected_vol_fractions, result.vol_fractions);
        EXPECT_SOFT_EQ(0, result.failed);
    }
    if (CELER_USE_DEVICE)
    {
        SCOPED_TRACE("Device heuristic");
        auto result = this->run_heuristic_init_device(num_tracks);
        EXPECT_VEC_SOFT_EQ(expected_vol_fractions, result.vol_fractions);
        EXPECT_SOFT_EQ(0, result.failed);
    }
}

//---------------------------------------------------------------------------//
} // namespace test
} // namespace celeritas
