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

        osi_.problem.physics.surfaces = [] {
            inp::SurfacePhysics input;

            // Center-top surface is only absorption

            PhysSurfaceId phys_surface{0};
            input.materials.push_back({});
            input.roughness.polished.emplace(phys_surface, inp::NoRoughness{});
            input.reflectivity.fresnel.emplace(phys_surface,
                                               inp::FresnelReflection{});
            input.interaction.trivial.emplace(
                phys_surface, optical::TrivialInteractionMode::absorb);

            // Default surface is transmission

            phys_surface++;
            input.materials.push_back({});
            input.roughness.polished.emplace(phys_surface, inp::NoRoughness{});
            input.reflectivity.fresnel.emplace(phys_surface,
                                               inp::FresnelReflection{});
            input.interaction.trivial.emplace(
                phys_surface, optical::TrivialInteractionMode::transmit);

            return input;
        }();

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
 * - Figure out how to add detectors in setup
 * - Write a test to check individual photon hit results
 * - Write a test to check bulk photon hits
 * - Write a test to check bulk photon hits on device
 */

TEST_F(DetectorTest, simple)
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
