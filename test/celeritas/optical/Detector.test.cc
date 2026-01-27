//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/Detector.test.cc
//---------------------------------------------------------------------------//
#include <utility>
#include <vector>

#include "geocel/UnitUtils.hh"
#include "celeritas/inp/StandaloneInput.hh"
#include "celeritas/optical/Runner.hh"
#include "celeritas/optical/Types.hh"
#include "celeritas/optical/detector/DetectorData.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
class DetectorTest : public Test
{
  public:
    void SetUp() override
    {
        osi_.problem.model.geometry
            = Test::test_data_path("geocel", "optical-box.gdml");

        osi_.problem.generator = inp::OpticalDirectGenerator{};
        osi_.problem.capacity = [] {
            inp::OpticalStateCapacity cap;
            cap.tracks = 32;
            cap.primaries = 8 * cap.tracks;
            cap.generators = 2 * cap.tracks;
            return cap;
        }();

        osi_.problem.num_streams = 1;

        osi_.geant_setup = GeantOpticalPhysicsOptions::deactivated();
        osi_.geant_setup.absorption = true;

        osi_.problem.model.detectors.detectors = {
            {"y-detectors", {VolumeId{2}}},
            {"x-detectors", {VolumeId{3}, VolumeId{4}}},
            {"z-detectors", {VolumeId{5}, VolumeId{6}}},
        };
    }

  protected:
    inp::OpticalStandaloneInput osi_;
};

/*
 * - Write a test to check bulk photon hits
 * - Write a test to check bulk photon hits on device
 */

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

    void operator()(Span<optical::DetectorHit> const& new_hits)
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
    osi_.problem.scoring.detector_callback = SimpleScorer{scores};

    using E = units::MevEnergy;
    using TI = optical::TrackInitializer;

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
           10,  // time
           {},
           ImplVolumeId{0}},
        TI{E{3e-6},
           Real3{0, 0, 0},  // pos
           Real3{0, 0, 1},  // dir
           Real3{0, 1, 0},  // pol
           1,  // time
           {},
           ImplVolumeId{0}},
        TI{E{4e-6},
           Real3{0, 0, 0},  // pos
           Real3{0, 0, -1},  // dir
           Real3{0, 1, 0},  // pol
           20,  // time
           {},
           ImplVolumeId{0}},
        TI{E{5e-6},
           Real3{0, 0, 0},  // pos
           Real3{1, 0, 0},  // dir
           Real3{0, 1, 0},  // pol
           13,  // time
           {},
           ImplVolumeId{0}},
        TI{E{6e-6},
           Real3{0, 0, 0},  // pos
           Real3{0, -1, 0},  // dir
           Real3{1, 0, 0},  // pol
           7,  // time
           {},
           ImplVolumeId{0}},
    };

    auto result = optical::Runner(std::move(osi_))(make_span(inits));

    EXPECT_EQ(6, result.counters.steps);
    EXPECT_EQ(1, result.counters.step_iters);
    EXPECT_EQ(1, result.counters.flushes);
    ASSERT_EQ(1, result.counters.generators.size());

    real_type const flight_time = 1.66782047599076e-09;

    static size_type const expected_detector_ids[] = {1, 1, 2, 2, 1, 0};
    static real_type const expected_energies[]
        = {1e-6, 2e-6, 3e-6, 4e-6, 5e-6, 6e-6};
    static real_type const expected_x_positions[] = {50, -50, 0, 0, 50, 0};
    static real_type const expected_y_positions[] = {0, 0, 0, 0, 0, -50};
    static real_type const expected_z_positions[] = {0, 0, 50, -50, 0, 0};
    static real_type const expected_times[] = {
        0 + flight_time,
        10 + flight_time,
        1 + flight_time,
        20 + flight_time,
        13 + flight_time,
        7 + flight_time,
    };
    static size_type const expected_volume_instance_ids[] = {5, 4, 6, 7, 5, 3};

    EXPECT_VEC_EQ(expected_detector_ids, scores.detector_ids);
    EXPECT_VEC_SOFT_EQ(expected_energies, scores.energies);
    EXPECT_VEC_SOFT_EQ(expected_x_positions, scores.x_positions);
    EXPECT_VEC_SOFT_EQ(expected_y_positions, scores.y_positions);
    EXPECT_VEC_SOFT_EQ(expected_z_positions, scores.z_positions);
    EXPECT_VEC_SOFT_EQ(expected_times, scores.times);
    EXPECT_VEC_EQ(expected_volume_instance_ids, scores.volume_instance_ids);
}

TEST_F(DetectorTest, stress)
{
    size_type num_tracks = osi_.problem.capacity.tracks * 4;

    std::vector<optical::TrackInitializer> const inits(
        num_tracks,
        optical::TrackInitializer{units::MevEnergy{3e-6},
                                  from_cm(Real3{0, 49, 0}),
                                  Real3{0, -1, 0},  // direction
                                  Real3{0, 0, 1},  // polarization
                                  0,
                                  {},  // primary
                                  ImplVolumeId{0}});

    auto result = optical::Runner(std::move(osi_))(make_span(inits));

    EXPECT_EQ(0, result.counters.steps);
    EXPECT_EQ(0, result.counters.step_iters);
    EXPECT_EQ(1, result.counters.flushes);
    ASSERT_EQ(1, result.counters.generators.size());

    auto const& gen = result.counters.generators.front();
    EXPECT_EQ(num_tracks, gen.buffer_size);
    EXPECT_EQ(0, gen.num_pending);
    EXPECT_EQ(num_tracks, gen.num_generated);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
