//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/univ/SimpleUnitTracker.test.cc
//---------------------------------------------------------------------------//
#include "orange/univ/SimpleUnitTracker.hh"

#include <algorithm>
#include <random>

#include "corecel/Config.hh"

#include "corecel/Assert.hh"
#include "corecel/data/CollectionAlgorithms.hh"
#include "corecel/data/CollectionStateStore.hh"
#include "corecel/data/Ref.hh"
#include "corecel/io/Repr.hh"
#include "corecel/math/ArrayUtils.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/Stopwatch.hh"
#include "geocel/random/IsotropicDistribution.hh"
#include "geocel/random/UniformBoxDistribution.hh"
#include "orange/OrangeGeoTestBase.hh"
#include "orange/OrangeParams.hh"
#include "orange/detail/UniverseIndexer.hh"

#include "SimpleUnitTracker.test.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
constexpr real_type sqrt_three{constants::sqrt_three};
constexpr real_type sqrt_two{constants::sqrt_two};
constexpr real_type sqrt_half = sqrt_two / real_type{2};

//---------------------------------------------------------------------------//
// TEST FIXTURES
//---------------------------------------------------------------------------//

class SimpleUnitTrackerTest : public OrangeGeoTestBase
{
  protected:
    using StateHostValue = HostVal<OrangeStateData>;
    using StateHostRef = HostRef<OrangeStateData>;
    using HostStateStore
        = CollectionStateStore<OrangeStateData, MemSpace::host>;
    using Initialization = ::celeritas::detail::Initialization;
    using LocalState = ::celeritas::detail::LocalState;

    struct HeuristicInitResult
    {
        std::vector<double> vol_fractions;  //!< Fraction per volume ID
        double failed{0};  //!< Fraction that couldn't initialize
        double walltime_per_track_ns{0};  //!< Kernel time

        void print_expected() const;
    };

  protected:
    // Initialization without any logical state
    LocalState make_state(Real3 pos, Real3 dir);

    // Initialization inside a volume
    LocalState make_state(Real3 pos, Real3 dir, char const* vol);

    // Initialization on a surface
    LocalState make_state(
        Real3 pos, Real3 dir, char const* vol, char const* surf, char sense);

    // Prepare for initialization across a surface
    LocalState make_state_crossing(
        Real3 pos, Real3 dir, char const* vol, char const* surf, char sense);

    HeuristicInitResult run_heuristic_init_host(size_type num_tracks) const;
    HeuristicInitResult run_heuristic_init_device(size_type num_tracks) const;

  private:
    StateHostValue setup_heuristic_states(size_type num_tracks) const;
    HeuristicInitResult
    reduce_heuristic_init(StateHostRef const&, double) const;
};

class DetailTest : public OrangeGeoTestBase
{
    void SetUp() override
    {
        TwoVolInput geo_inp;
        geo_inp.radius = 1.5;
        this->build_geometry(geo_inp);
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

#define FieldLayersTest FieldLayersTest
class FieldLayersTest : public SimpleUnitTrackerTest
{
    void SetUp() override { this->build_geometry("field-layers.org.json"); }
};

#define FiveVolumesTest FiveVolumesTest
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
SimpleUnitTrackerTest::make_state(Real3 pos, Real3 dir, char const* vol)
{
    LocalState state = this->make_state(pos, dir);
    detail::UniverseIndexer ui(this->host_params().universe_indexer_data);
    state.volume = vol ? ui.local_volume(this->find_volume(vol)).volume
                       : LocalVolumeId{};
    return state;
}

//---------------------------------------------------------------------------//
/*!
 * Initialize crossing a surface.
 *
 * This takes the *before-crossing volume* and *before-crossing sense*.
 */
LocalState SimpleUnitTrackerTest::make_state(
    Real3 pos, Real3 dir, char const* vol, char const* surf, char sense)
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
    detail::UniverseIndexer ui(this->host_params().universe_indexer_data);
    state.volume = ui.local_volume(this->find_volume(vol)).volume;
    // *Intentionally* flip the sense because we're looking for the
    // post-crossing volume. This is normally done by the multi-level
    // TrackingGeometry.
    state.surface = {ui.local_surface(this->find_surface(surf)).surface,
                     before_crossing_sense};
    return state;
}

//---------------------------------------------------------------------------//
/*!
 * Initialize crossing a surface.
 *
 * This takes the *before-crossing volume* and *before-crossing sense*.
 */
LocalState SimpleUnitTrackerTest::make_state_crossing(
    Real3 pos, Real3 dir, char const* vol, char const* surf, char sense)
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
    InitializingExecutor<> calc_init{this->host_params(), states.ref()};

    // Loop over all track slots
    Stopwatch get_time;
    for (auto tid : range(TrackSlotId{states.size()}))
    {
        calc_init(tid);
    }

    return this->reduce_heuristic_init(states.ref(), get_time());
}

//---------------------------------------------------------------------------//
/*!
 * Initialize particles randomly and tally their resulting locations.
 */
auto SimpleUnitTrackerTest::run_heuristic_init_device(
    size_type num_tracks) const -> HeuristicInitResult
{
    using DStateStore = CollectionStateStore<OrangeStateData, MemSpace::device>;
    DStateStore states(this->setup_heuristic_states(num_tracks));

    // Run on device
    Stopwatch get_time;
    test_initialize(this->params().device_ref(), states.ref());
    double const kernel_time = get_time();

    // Copy result back to host
    HostStateStore state_host(states.ref());
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
    resize(&result, this->host_params(), num_tracks);
    auto result_ref = make_ref(result);

    std::mt19937 rng;

    // Sample uniform in space and isotropic in direction
    auto const& bbox = this->params().bbox();
    UniformBoxDistribution<> sample_box{bbox.lower(), bbox.upper()};
    IsotropicDistribution<> sample_isotropic;
    for (auto i : range(num_tracks))
    {
        auto lsa = LevelStateAccessor(&result_ref, TrackSlotId{i}, LevelId{0});
        lsa.pos() = sample_box(rng);
        lsa.dir() = sample_isotropic(rng);
    }

    // Clear other data
    fill(LocalVolumeId{}, &result.vol);
    fill(LocalSurfaceId{}, &result.surf);
    fill(LevelId{}, &result.level);

    CELER_ENSURE(result);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Process "heuristic init" test results.
 */
auto SimpleUnitTrackerTest::reduce_heuristic_init(
    StateHostRef const& host, double wall_time) const -> HeuristicInitResult
{
    CELER_EXPECT(host);
    CELER_EXPECT(wall_time > 0);
    CELER_EXPECT(this->geometry()->max_depth() == 1);
    std::vector<size_type> counts(this->num_volumes());
    size_type error_count{};

    for (auto i : range(host.size()))
    {
        auto tid = TrackSlotId{i};
        LevelStateAccessor lsa(&host, tid, LevelId{0});
        auto vol = lsa.vol();

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
    double const norm = 1.0 / static_cast<double>(host.size());
    for (auto i : range(counts.size()))
    {
        result.vol_fractions[i] = norm * static_cast<double>(counts[i]);
    }
    result.failed = norm * error_count;
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

TEST_F(DetailTest, bumpcalculator)
{
    detail::BumpCalculator calc_bump(
        Tolerance<>::from_relative(1e-8, /* length = */ 0.1));
    EXPECT_SOFT_EQ(1e-9, calc_bump(Real3{0, 0, 0}));
    EXPECT_SOFT_EQ(1e-9, calc_bump(Real3{1e-14, 0, 0}));
    EXPECT_SOFT_EQ(2e-8, calc_bump(Real3{0, 1, 2}));
    EXPECT_SOFT_EQ(1e-6, calc_bump(Real3{-100, 1, 2}));
    EXPECT_SOFT_EQ(1e-2, calc_bump(Real3{0, 0, 1e6}));
    EXPECT_SOFT_EQ(1e1, calc_bump(Real3{0, 1e9, 1e6}));
}

TEST_F(OneVolumeTest, initialize)
{
    SimpleUnitTracker tracker(this->host_params(), SimpleUnitId{0});

    {
        // Anywhere is valid
        auto init = tracker.initialize(this->make_state({1, 2, 3}, {0, 0, 1}));
        EXPECT_TRUE(init);
        EXPECT_EQ(LocalVolumeId{0}, init.volume);
        EXPECT_FALSE(init.surface);
    }
}

TEST_F(OneVolumeTest, intersect)
{
    SimpleUnitTracker tracker(this->host_params(), SimpleUnitId{0});

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
    SimpleUnitTracker tracker(this->host_params(), SimpleUnitId{0});
    detail::UniverseIndexer ui(this->host_params().universe_indexer_data);

    EXPECT_SOFT_EQ(
        inf,
        tracker.safety({1, 2, 3},
                       ui.local_volume(this->find_volume("infinite")).volume));
}

TEST_F(OneVolumeTest, heuristic_init)
{
    size_type num_tracks = 1024;
    static double const expected_vol_fractions[] = {1.0};

    {
        SCOPED_TRACE("Host heuristic");
        auto result = this->run_heuristic_init_host(num_tracks);

        EXPECT_VEC_SOFT_EQ(expected_vol_fractions, result.vol_fractions);
        EXPECT_SOFT_EQ(0, result.failed);
    }

    if (celeritas::device())
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
    SimpleUnitTracker tracker(this->host_params(), SimpleUnitId{0});

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
    SimpleUnitTracker tracker(this->host_params(), SimpleUnitId{0});

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
    SimpleUnitTracker tracker(this->host_params(), SimpleUnitId{0});

    {
        SCOPED_TRACE("Inside");
        auto state = this->make_state({0.5, 0, 0}, {0, 0, 1}, "inside");
        auto isect = tracker.intersect(state);
        EXPECT_TRUE(isect);
        EXPECT_EQ(LocalSurfaceId{0}, isect.surface.id());
        EXPECT_EQ(Sense::inside, isect.surface.unchecked_sense());
        EXPECT_SOFT_EQ(sqrt_two, isect.distance);

        state = this->make_state({0.5, 0, 0}, {1, 0, 0}, "inside");
        isect = tracker.intersect(state);
        EXPECT_TRUE(isect);
        EXPECT_EQ(LocalSurfaceId{0}, isect.surface.id());
        EXPECT_EQ(Sense::inside, isect.surface.unchecked_sense());
        EXPECT_SOFT_EQ(1.0, isect.distance);

        // Range limit: further than surface
        isect = tracker.intersect(state, 10.0);
        EXPECT_TRUE(isect);
        EXPECT_EQ(LocalSurfaceId{0}, isect.surface.id());
        EXPECT_EQ(Sense::inside, isect.surface.unchecked_sense());
        EXPECT_SOFT_EQ(1.0, isect.distance);

        // Coincident
        isect = tracker.intersect(state, 1.0);
        EXPECT_TRUE(isect);
        EXPECT_EQ(LocalSurfaceId{0}, isect.surface.id());
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
        EXPECT_EQ(LocalSurfaceId{0}, isect.surface.id());
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
        EXPECT_EQ(LocalSurfaceId{0}, isect.surface.id());
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
        EXPECT_EQ(LocalSurfaceId{0}, isect.surface.id());
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
        EXPECT_EQ(LocalSurfaceId{0}, isect.surface.id());
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
    SimpleUnitTracker tracker(this->host_params(), SimpleUnitId{0});
    detail::UniverseIndexer ui(this->host_params().universe_indexer_data);
    LocalVolumeId outside
        = ui.local_volume(this->find_volume("outside")).volume;
    LocalVolumeId inside = ui.local_volume(this->find_volume("inside")).volume;

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
    EXPECT_SOFT_EQ(inf, tracker.safety({0, 0, 0}, inside));  // degenerate!
#endif
}

TEST_F(TwoVolumeTest, normal)
{
    SimpleUnitTracker tracker(this->host_params(), SimpleUnitId{0});

    if (CELERITAS_DEBUG)
    {
        SCOPED_TRACE("Not on a surface");
        EXPECT_THROW(tracker.normal(Real3{0, 0, 1.6}, LocalSurfaceId{}),
                     DebugError);
    }
    {
        Real3 pos{3, -2, 1};
        Real3 expected_normal;
        auto invnorm = 1 / norm(pos);
        for (auto i : range(3))
        {
            expected_normal[i] = pos[i] * invnorm;
            pos[i] = expected_normal[i] * real_type(1.5);  // radius
        }

        auto actual_normal = tracker.normal(pos, LocalSurfaceId{0});
        EXPECT_VEC_SOFT_EQ(expected_normal, actual_normal);
    }
}

TEST_F(TwoVolumeTest, TEST_IF_CELERITAS_DOUBLE(heuristic_init))
{
    size_type num_tracks = 1024;

    static double const expected_vol_fractions[] = {0.4765625, 0.5234375};

    {
        SCOPED_TRACE("Host heuristic");
        auto result = this->run_heuristic_init_host(num_tracks);

        EXPECT_VEC_SOFT_EQ(expected_vol_fractions, result.vol_fractions);
        EXPECT_SOFT_EQ(0, result.failed);
    }
    if (celeritas::device())
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
    SimpleUnitTracker tracker(this->host_params(), SimpleUnitId{0});

    {
        SCOPED_TRACE("Exterior");
        auto init
            = tracker.initialize(this->make_state({0, 50, 0}, {0, -1, 0}));
        EXPECT_EQ("[EXTERIOR]", this->id_to_label(init.volume));
        EXPECT_FALSE(init.surface);
    }
    {
        auto init = tracker.initialize(this->make_state({0, -3, 0}, {0, 0, 1}));
        EXPECT_EQ("world.bg", this->id_to_label(init.volume));
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
    SimpleUnitTracker tracker(this->host_params(), SimpleUnitId{0});

    // Test crossing into and out of the "background" with varying levels
    // of numerical imprecision
    for (real_type eps : {-1e-6, -1e-10, -1e-14, 0.0, 1e-14, 1e-10, 1e-6})
    {
        SCOPED_TRACE(eps);
        {
            // From background to volume
            auto init = tracker.cross_boundary(
                this->make_state_crossing({0, real_type{-1.5} + eps, 0},
                                          {0, -1, 0},
                                          "world.bg",
                                          "layerbox1.py",
                                          '+'));
            EXPECT_EQ("layer1", this->id_to_label(init.volume));
            EXPECT_EQ("layerbox1.py", this->id_to_label(init.surface.id()));
            EXPECT_EQ(Sense::inside, init.surface.unchecked_sense());
        }
        {
            // From volume to background
            auto init = tracker.cross_boundary(
                this->make_state_crossing({0, real_type{-2.5} - eps, 0},
                                          {0, -1, 0},
                                          "layer1",
                                          "layerbox1.my",
                                          '+'));
            EXPECT_EQ("world.bg", this->id_to_label(init.volume));
            EXPECT_EQ("layerbox1.my", this->id_to_label(init.surface.id()));
            EXPECT_EQ(Sense::inside, init.surface.unchecked_sense());
        }
    }
}

TEST_F(FieldLayersTest, intersect)
{
    SimpleUnitTracker tracker(this->host_params(), SimpleUnitId{0});

    {
        SCOPED_TRACE("straightforward");
        auto state = this->make_state({0, -1, 0}, {0, 1, 0}, "world.bg");
        auto isect = tracker.intersect(state);
        EXPECT_TRUE(isect);
        EXPECT_EQ("layerbox2.my", this->id_to_label(isect.surface.id()));
        EXPECT_EQ(Sense::inside, isect.surface.unchecked_sense());
        EXPECT_SOFT_EQ(0.5, isect.distance);
    }
    {
        SCOPED_TRACE("crossing internal planes");
        auto state = this->make_state({9.6, 4, 9.7}, {-1, -1, -1}, "world.bg");
        auto isect = tracker.intersect(state);
        EXPECT_TRUE(isect);
        EXPECT_EQ("layerbox3.py", this->id_to_label(isect.surface.id()));
        EXPECT_EQ(Sense::outside, isect.surface.unchecked_sense());
        EXPECT_SOFT_EQ(1.5 * sqrt_three, isect.distance);
    }
}

TEST_F(FieldLayersTest, TEST_IF_CELERITAS_DOUBLE(heuristic_init))
{
    size_type num_tracks = 8192;
    static double const expected_vol_fractions[] = {0,
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

    if (celeritas::device())
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
    auto const& bbox = this->params().bbox();
    EXPECT_VEC_SOFT_EQ(Real3({-1.5, -1.5, -0.5}), bbox.lower());
    EXPECT_VEC_SOFT_EQ(Real3({1.5, 1.5, 0.5}), bbox.upper());
}

TEST_F(FiveVolumesTest, initialize)
{
    SimpleUnitTracker tracker(this->host_params(), SimpleUnitId{0});
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
    SimpleUnitTracker tracker(this->host_params(), SimpleUnitId{0});

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
        real_type eps = 1e-10;
        auto init = tracker.cross_boundary(this->make_state_crossing(
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
    SimpleUnitTracker tracker(this->host_params(), SimpleUnitId{0});

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
    SimpleUnitTracker tracker(this->host_params(), SimpleUnitId{0});
    detail::UniverseIndexer ui(this->host_params().universe_indexer_data);
    LocalVolumeId a = ui.local_volume(this->find_volume("a")).volume;
    LocalVolumeId d = ui.local_volume(this->find_volume("d")).volume;

    EXPECT_SOFT_EQ(0.15138781886599728, tracker.safety({-0.75, 0.5, 0}, a));

    // Note: distance with "d" surfaces is conservatively to internal planes
    EXPECT_SOFT_EQ(0.5, tracker.safety({-5, 20, 0}, d));
}

TEST_F(FiveVolumesTest, TEST_IF_CELERITAS_DOUBLE(heuristic_init))
{
    size_type num_tracks = 10000;

    static double const expected_vol_fractions[]
        = {0, 0.0701, 0.106, 0.1621, 0.6555, 0.0063};

    {
        SCOPED_TRACE("Host heuristic");
        auto result = this->run_heuristic_init_host(num_tracks);
        EXPECT_VEC_SOFT_EQ(expected_vol_fractions, result.vol_fractions);
        EXPECT_SOFT_EQ(0, result.failed);
    }
    if (celeritas::device())
    {
        SCOPED_TRACE("Device heuristic");
        auto result = this->run_heuristic_init_device(num_tracks);
        EXPECT_VEC_SOFT_EQ(expected_vol_fractions, result.vol_fractions);
        EXPECT_SOFT_EQ(0, result.failed);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
