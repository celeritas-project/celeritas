//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/Detector.test.cc
//---------------------------------------------------------------------------//
#include <numeric>
#include <utility>
#include <vector>

#include "geocel/UnitUtils.hh"
#include "geocel/VolumeParams.hh"
#include "geocel/inp/Model.hh"
#include "celeritas/GeantTestBase.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/inp/StandaloneInput.hh"
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/DetectorData.hh"
#include "celeritas/optical/Runner.hh"
#include "celeritas/optical/Transporter.hh"
#include "celeritas/optical/Types.hh"
#include "celeritas/optical/gen/DirectGeneratorAction.hh"
#include "celeritas/optical/gen/PrimaryGeneratorAction.hh"
#include "celeritas/optical/surface/SurfacePhysicsParams.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace optical
{
namespace test
{
using namespace ::celeritas::test;

constexpr bool reference_configuration
    = ((CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
       && CELERITAS_CORE_RNG == CELERITAS_CORE_RNG_XORWOW);

//---------------------------------------------------------------------------//
/*!
 * Test optical detector and scoring.
 *
 * Most physics is disabled to get a simple photon to hit mapping.
 */
class DetectorTest : public Test
{
  public:
    void SetUp() override
    {
        // Set per-process state sizes
        osi_.problem.capacity = [] {
            inp::OpticalStateCapacity cap;
            cap.tracks = 4096;
            cap.primaries = 8 * cap.tracks;
            cap.generators = 2 * cap.tracks;
            return cap;
        }();

        // Run on a single stream
        osi_.problem.num_streams = 1;

        // Set optical physics processes
        osi_.geant_setup = [] {
            GeantOpticalPhysicsOptions opt;
            opt.cherenkov = std::nullopt;
            opt.scintillation = std::nullopt;
            opt.wavelength_shifting = std::nullopt;
            opt.wavelength_shifting2 = std::nullopt;
            opt.rayleigh_scattering = false;
            opt.mie_scattering = false;
            // absorption and boundary remain enabled (defaults)
            return opt;
        }();

        // Set optical detectors to load from GDML
        osi_.detectors = {"x-detectors", "y-detectors", "z-detectors"};
    }

  protected:
    inp::OpticalStandaloneInput osi_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
// Run test to check small number of photons and hits to ensure correct hit
// information is populated.

struct SimpleScores
{
    std::vector<size_type> detector_ids;
    std::vector<real_type> energies;
    std::vector<real_type> times;
    std::vector<real_type> x_positions;
    std::vector<real_type> y_positions;
    std::vector<real_type> z_positions;
    std::vector<size_type> volume_instance_ids;
};

struct SimpleScorer
{
    SimpleScores& scores;

    void operator()(Span<DetectorHit const> new_hits)
    {
        for (auto const& hit : new_hits)
        {
            scores.detector_ids.push_back(hit.detector.unchecked_get());
            scores.energies.push_back(value_as<units::MevEnergy>(hit.energy));
            scores.times.push_back(hit.time);
            scores.x_positions.push_back(hit.position[0]);
            scores.y_positions.push_back(hit.position[1]);
            scores.z_positions.push_back(hit.position[2]);
            scores.volume_instance_ids.push_back(
                hit.volume_instance.unchecked_get());
        }
    }
};

TEST_F(DetectorTest, simple)
{
    SimpleScores scores;
    osi_.problem.detectors.callback = SimpleScorer{scores};

    // Manually generate arbitrary photons aimed at different detectors

    using E = units::MevEnergy;
    using TI = TrackInitializer;

    std::vector<TI> const inits{
        TI{E{1e-6},
           Real3{0, 0, 0},  // pos
           Real3{1, 0, 0},  // dir
           Real3{0, 1, 0},  // pol
           0,  // time
           {},
           ImplVolumeId{0}},
        TI{E{2e-6},
           Real3{0, 0, 0},  // pos
           Real3{-1, 0, 0},  // dir
           Real3{0, 1, 0},  // pol
           0,  // time
           {},
           ImplVolumeId{0}},
        TI{E{3e-6},
           Real3{0, 0, 0},  // pos
           Real3{0, 0, 1},  // dir
           Real3{0, 1, 0},  // pol
           0,  // time
           {},
           ImplVolumeId{0}},
        TI{E{4e-6},
           Real3{0, 0, 0},  // pos
           Real3{0, 0, -1},  // dir
           Real3{0, 1, 0},  // pol
           0,  // time
           {},
           ImplVolumeId{0}},
        TI{E{5e-6},
           Real3{0, 0, 0},  // pos
           Real3{1, 0, 0},  // dir
           Real3{0, 1, 0},  // pol
           0,  // time
           {},
           ImplVolumeId{0}},
        TI{E{6e-6},
           Real3{0, 0, 0},  // pos
           Real3{0, -1, 0},  // dir
           Real3{1, 0, 0},  // pol
           0,  // time
           {},
           ImplVolumeId{0}},
        TI{E{2e-7},
           Real3{0, 0, 0},  // pos
           Real3{1, 0, 0},  // dir
           Real3{0, 1, 0},  // pol
           0,  // time
           {},
           ImplVolumeId{0}},
    };

    // Set geometry filename
    osi_.problem.model.geometry
        = Test::test_data_path("geocel", "optical-box-det-tra.gdml");

    // Create direct generator input
    osi_.problem.generator = celeritas::inp::OpticalDirectGenerator{};

    // Construct the runner and transport optical primaries
    optical::Runner run(std::move(osi_));
    run.insert(make_span(std::as_const(inits)));
    run();

    // Check results

    real_type const box_size = from_cm(50);
    real_type const flight_time = box_size / constants::c_light;

    static size_type const expected_detector_ids[] = {1, 1, 2, 2, 1, 0, 1};
    static real_type const expected_energies[]
        = {1e-6, 2e-6, 3e-6, 4e-6, 5e-6, 6e-6, 2e-07};
    static real_type const expected_x_positions[] = {
        box_size,
        -box_size,
        0,
        0,
        box_size,
        0,
        box_size,
    };
    static real_type const expected_y_positions[] = {
        0,
        0,
        0,
        0,
        0,
        -box_size,
        0,
    };
    static real_type const expected_z_positions[] = {
        0,
        0,
        box_size,
        -box_size,
        0,
        0,
        0,
    };
    // adjusted by group velocity
    static double const expected_times[] = {
        1.49995 * flight_time,
        1.3333 * flight_time,
        3.66675 * flight_time,
        2 * flight_time,
        2 * flight_time,
        2 * flight_time,
        flight_time,
    };

    static size_type const expected_volume_instance_ids[]
        = {5, 4, 6, 7, 5, 3, 5};

    if (reference_configuration)
    {
        EXPECT_VEC_EQ(expected_detector_ids, scores.detector_ids);
        EXPECT_VEC_SOFT_EQ(expected_energies, scores.energies);
        EXPECT_VEC_SOFT_EQ(expected_x_positions, scores.x_positions);
        EXPECT_VEC_SOFT_EQ(expected_y_positions, scores.y_positions);
        EXPECT_VEC_SOFT_EQ(expected_z_positions, scores.z_positions);
        EXPECT_VEC_SOFT_EQ(expected_times, scores.times);
        EXPECT_VEC_EQ(expected_volume_instance_ids, scores.volume_instance_ids);
    }
}

//---------------------------------------------------------------------------//
// Run test over large number of photons to check buffering is done correctly.

struct StressScorer
{
    std::vector<size_type>& scores;
    size_type& errored;

    void operator()(Span<DetectorHit const> hits)
    {
        for (auto const& hit : hits)
        {
            if (hit.detector < scores.size())
            {
                scores[hit.detector.get()]++;
            }
            else
            {
                errored++;
            }
        }
    }
};

TEST_F(DetectorTest, stress)
{
    // 3 detectors: x, y, z
    std::vector<size_type> hits(3, 0);
    size_type errored = 0;
    osi_.problem.detectors.callback = StressScorer{hits, errored};

    // Set geometry filename
    osi_.problem.model.geometry
        = Test::test_data_path("geocel", "optical-box-det-tra.gdml");

    // Isotropically generate photons
    osi_.problem.generator = [] {
        inp::OpticalPrimaryGenerator gen;
        gen.primaries = 8192;
        gen.energy = inp::MonoenergeticDistribution{1e-5};
        gen.angle = inp::IsotropicDistribution{};
        gen.shape = inp::PointDistribution{{0, 0, 0}};
        return gen;
    }();

    // Construct the runner and transport optical primaries
    optical::Runner run(std::move(osi_));
    run.insert();
    run();

    // Check results

    if (reference_configuration)
    {
        static size_type const expected_hits[] = {2743, 2716, 2733};

        EXPECT_VEC_EQ(expected_hits, hits);
        EXPECT_EQ(errored, 0);
    }
}

//---------------------------------------------------------------------------//
// Test surface efficiency propagates hits to detector
TEST_F(DetectorTest, efficiency)
{
    // 3 detectors: x, y, z
    std::vector<size_type> hits(3, 0);
    size_type errored = 0;
    osi_.problem.detectors.callback = StressScorer{hits, errored};

    // Set geometry filename
    osi_.problem.model.geometry
        = Test::test_data_path("geocel", "optical-box-det-eff.gdml");

    // Isotropically generate photons
    osi_.problem.generator = [] {
        inp::OpticalPrimaryGenerator gen;
        gen.primaries = 8192;
        gen.energy = inp::MonoenergeticDistribution{1e-5};
        gen.angle = inp::IsotropicDistribution{};
        gen.shape = inp::PointDistribution{{0, 0, 0}};
        return gen;
    }();

    // Construct the runner and transport optical primaries
    optical::Runner run(std::move(osi_));
    run.insert();
    run();

    // Check results
    if constexpr (reference_configuration)
    {
        static size_type const expected_hits[] = {2776, 2713, 2703};

        EXPECT_VEC_EQ(expected_hits, hits);
        EXPECT_EQ(errored, 0);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
