//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SimpleUnitTracker.test.cc
//---------------------------------------------------------------------------//
#include "orange/universes/SimpleUnitTracker.hh"

#include <random>

// Source includes
#include "base/ArrayUtils.hh"
#include "base/Constants.hh"
#include "base/Repr.hh"
#include "base/Stopwatch.hh"

// Test includes
#include "celeritas_test.hh"
#include "orange/OrangeGeoTestBase.hh"
#include "random/distributions/IsotropicDistribution.hh"
#include "random/distributions/UniformBoxDistribution.hh"
#include "SimpleUnitTracker.test.hh"

using namespace celeritas;
using namespace celeritas_test;
using celeritas::constants::sqrt_two;
using celeritas::detail::Initialization;
using celeritas::detail::LocalState;

namespace
{
constexpr real_type sqrt_half = sqrt_two / 2;
}

//---------------------------------------------------------------------------//
// TEST FIXTURES
//---------------------------------------------------------------------------//

class SimpleUnitTrackerTest : public celeritas_test::OrangeGeoTestBase
{
  protected:
    using StateHostValue
        = celeritas::OrangeStateData<Ownership::value, MemSpace::host>;

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

    // Initialization crossing a surface
    LocalState make_state(
        Real3 pos, Real3 dir, const char* vol, const char* surf, char sense);

    HeuristicInitResult run_heuristic_init_host(size_type num_tracks) const;
    HeuristicInitResult run_heuristic_init_device(size_type num_tracks) const;

  private:
    StateHostValue      setup_heuristic_states(size_type num_tracks) const;
    HeuristicInitResult reduce_heuristic_init(StateHostValue, double) const;
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

//! Construct a test name that is disabled when JSON is disabled
#if CELERITAS_USE_JSON
#    define TEST_IF_CELERITAS_JSON(name) name
#else
#    define TEST_IF_CELERITAS_JSON(name) DISABLED_##name
#endif

#define FiveVolumesTest TEST_IF_CELERITAS_JSON(FiveVolumesTest)
class FiveVolumesTest : public SimpleUnitTrackerTest
{
    void SetUp() override
    {
        if (!CELERITAS_USE_JSON)
        {
            GTEST_SKIP() << "JSON is not enabled";
        }

        this->build_geometry("five-volumes.org.json");
    }
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
    state.pos         = pos;
    state.dir         = dir;
    state.volume      = {};
    state.surface     = {};
    state.temp_senses = this->sense_storage();
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
    state.surface
        = {this->find_surface(surf), flip_sense(before_crossing_sense)};
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
    auto state_host = this->setup_heuristic_states(num_tracks);

    // Set up for host run
    OrangeStateData<Ownership::reference, MemSpace::host> host_state_ref;
    host_state_ref = state_host;
    InitializingLauncher<> calc_init{this->params_host_ref(), host_state_ref};

    // Loop over all threads
    Stopwatch get_time;
    for (auto tid : range(ThreadId{state_host.size()}))
    {
        calc_init(tid);
    }

    return this->reduce_heuristic_init(std::move(state_host), get_time());
}

//---------------------------------------------------------------------------//
/*!
 * Initialize particles randomly and tally their resulting locations.
 */
auto SimpleUnitTrackerTest::run_heuristic_init_device(size_type num_tracks) const
    -> HeuristicInitResult
{
    // Construct on host and copy to device
    auto state_host = this->setup_heuristic_states(num_tracks);
    OrangeStateData<Ownership::value, MemSpace::device> state_device;
    state_device = state_host;
    StateDeviceRef state_device_ref;
    state_device_ref = state_device;

    // Run on device
    Stopwatch get_time;
    test_initialize(this->params_device_ref(), state_device_ref);
    const double kernel_time = get_time();

    // Copy result back to host
    state_host = state_device;
    return this->reduce_heuristic_init(std::move(state_host), kernel_time);
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
    resize(&result, this->params_host_ref(), num_tracks);
    auto pos_view = result.pos[AllItems<Real3>{}];
    auto dir_view = result.dir[AllItems<Real3>{}];

    std::mt19937 rng;

    // Sample uniform in space and isotropic in direction
    UniformBoxDistribution<> sample_box{this->bbox_lower(), this->bbox_upper()};
    IsotropicDistribution<>  sample_isotropic;
    for (auto i : range(num_tracks))
    {
        pos_view[i] = sample_box(rng);
        dir_view[i] = sample_isotropic(rng);
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Process "heuristic init" test results.
 */
auto SimpleUnitTrackerTest::reduce_heuristic_init(StateHostValue host,
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

TEST_F(OneVolumeTest, initialize)
{
    SimpleUnitTracker tracker(this->params_host_ref());

    {
        // Anywhere is valid
        auto init = tracker.initialize(this->make_state({1, 2, 3}, {0, 0, 1}));
        EXPECT_TRUE(init);
        EXPECT_EQ(VolumeId{0}, init.volume);
        EXPECT_FALSE(init.surface);
    }
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
    if (CELERITAS_USE_CUDA)
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
    SimpleUnitTracker tracker(this->params_host_ref());

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
        SCOPED_TRACE("Crossing the boundary from the inside");
        auto init = tracker.initialize(
            this->make_state({1.5, 0, 0}, {0, 0, 1}, "inside", "sphere", '-'));
        EXPECT_EQ("outside", this->id_to_label(init.volume));
        EXPECT_EQ("sphere", this->id_to_label(init.surface.id()));
        EXPECT_EQ(Sense::outside, init.surface.sense());
    }
    {
        SCOPED_TRACE("Crossing the boundary from the outside");
        auto init = tracker.initialize(this->make_state(
            {1.5, 0, 0}, {0, 0, 1}, "outside", "sphere", '+'));
        EXPECT_EQ("inside", this->id_to_label(init.volume));
        EXPECT_EQ("sphere", this->id_to_label(init.surface.id()));
        EXPECT_EQ(Sense::inside, init.surface.sense());
    }
    {
        SCOPED_TRACE("Outside the sphere");
        auto init
            = tracker.initialize(this->make_state({3.0, 0, 0}, {0, 0, 1}));
        EXPECT_EQ("outside", this->id_to_label(init.volume));
        EXPECT_FALSE(init.surface);
    }
}

TEST_F(TwoVolumeTest, heuristic_init)
{
    size_type num_tracks = 1024;

    static const double expected_vol_fractions[] = {0.5234375, 0.4765625};

    {
        SCOPED_TRACE("Host heuristic");
        auto result = this->run_heuristic_init_host(num_tracks);

        EXPECT_VEC_SOFT_EQ(expected_vol_fractions, result.vol_fractions);
        EXPECT_SOFT_EQ(0, result.failed);
    }
    if (CELERITAS_USE_CUDA)
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
    EXPECT_VEC_SOFT_EQ(Real3({-1.5, -1.5, -0.5}), bbox_lower());
    EXPECT_VEC_SOFT_EQ(Real3({1.5, 1.5, 0.5}), bbox_upper());
}

TEST_F(FiveVolumesTest, initialize)
{
    SimpleUnitTracker tracker(this->params_host_ref());

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
    {
        SCOPED_TRACE("Crossing the boundary from the inside of 'e'");
        auto init = tracker.initialize(this->make_state(
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
        auto      init = tracker.initialize(this->make_state(
            {eps, -0.25, 0}, {1, 0, 0}, "e", "epsilon.s", '-'));
        EXPECT_EQ("c", this->id_to_label(init.volume));
        EXPECT_EQ("epsilon.s", this->id_to_label(init.surface.id()));
        EXPECT_EQ(Sense::outside, init.surface.sense());
    }
    {
        SCOPED_TRACE("Crossing the boundary into a more complicated region");
        auto init = tracker.initialize(
            this->make_state({-.75 * sqrt_half, 0.75 * sqrt_half, 0},
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
        auto init = tracker.initialize(
            this->make_state({-.75 * sqrt_half, 0.75 * sqrt_half, 0},
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
        // boundary. We'll either need to ensure that's ei
        SCOPED_TRACE("Crossing at triple point");
        auto init = tracker.initialize(
            this->make_state({0, 0.75, 0}, {0, 1, 0}, "c", "gamma.s", '-'));
        EXPECT_EQ("d", this->id_to_label(init.volume));
        EXPECT_EQ("gamma.s", this->id_to_label(init.surface.id()));
        EXPECT_EQ(Sense::outside, init.surface.sense());

        init = tracker.initialize(
            this->make_state({0, 0.75, 0}, {-1, 0, 0}, "d", "gamma.s", '+'));
        EXPECT_EQ("c", this->id_to_label(init.volume));
        EXPECT_EQ("gamma.s", this->id_to_label(init.surface.id()));
        EXPECT_EQ(Sense::inside, init.surface.sense());

        // Near triple point, on sphere but crossing plane edge
        init = tracker.initialize(
            this->make_state({0, 0.75, 0}, {-1, 0, 0}, "d", "alpha.px", '+'));
        EXPECT_EQ("a", this->id_to_label(init.volume));
        EXPECT_EQ("alpha.px", this->id_to_label(init.surface.id()));
        EXPECT_EQ(Sense::inside, init.surface.sense());
    }
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
    if (CELERITAS_USE_CUDA)
    {
        SCOPED_TRACE("Device heuristic");
        auto result = this->run_heuristic_init_device(num_tracks);
        EXPECT_VEC_SOFT_EQ(expected_vol_fractions, result.vol_fractions);
        EXPECT_SOFT_EQ(0, result.failed);
    }
}
