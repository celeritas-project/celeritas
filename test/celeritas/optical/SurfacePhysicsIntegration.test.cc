//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/SurfacePhysicsIntegration.test.cc
//---------------------------------------------------------------------------//

#include "SurfacePhysicsIntegration.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace optical
{
namespace test
{
using namespace ::celeritas::test;
//---------------------------------------------------------------------------//
/*!
 * Counters for photon status after a run at a single angle.
 */
struct CollectResults
{
    size_type num_absorbed{0};
    size_type num_failed{0};
    size_type num_reflected{0};
    size_type num_refracted{0};

    //! Clear counters
    void reset()
    {
        num_absorbed = 0;
        num_failed = 0;
        num_reflected = 0;
        num_refracted = 0;
    }

    //! Score track
    void operator()(CoreTrackView const& track)
    {
        if (track.sim().status() == TrackStatus::alive)
        {
            auto vol = track.geometry().volume_instance_id();
            if (vol == VolumeInstanceId{1})
            {
                num_reflected++;
                return;
            }
            else if (vol == VolumeInstanceId{2})
            {
                num_refracted++;
                return;
            }
        }
        else if (track.sim().status() == TrackStatus::killed)
        {
            num_absorbed++;
            return;
        }

        num_failed++;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Counter results for a series of runs at different angles.
 */
struct SurfaceTestResults
{
    std::vector<size_type> num_absorbed;
    std::vector<size_type> num_reflected;
    std::vector<size_type> num_refracted;
};

//---------------------------------------------------------------------------//
// TEST CHASSIS
//---------------------------------------------------------------------------//

class SurfacePhysicsIntegrationTest : public SurfacePhysicsIntegrationTestBase
{
  public:
    SurfaceTestResults run(std::vector<real_type> const& angles)
    {
        // Create collector
        auto& reg = *this->optical_params()->action_reg();
        auto collector = std::make_shared<CollectResultsAction<CollectResults>>(
            reg.next_id(), collect_);
        reg.insert(collector);

        this->initialize_run();

        // Run over angles
        SurfaceTestResults results;
        for (auto deg_angle : angles)
        {
            collect_.reset();

            this->run_step(deg_angle * constants::pi / 180);

            EXPECT_EQ(0, collect_.num_failed);
            results_.num_absorbed.push_back(collect_.num_absorbed);
            results_.num_reflected.push_back(collect_.num_reflected);
            results_.num_refracted.push_back(collect_.num_refracted);
        }

        return results;
    }

    void reference_run(std::vector<real_type> const& angles,
                       SurfaceTestResults const& expected)
    {
        if (reference_configuration)
        {
            auto result = this->run(angles);
            EXPECT_EQ(expected.num_reflected, result.num_reflected);
            EXPECT_EQ(expected.num_refracted, result.num_refracted);
            EXPECT_EQ(expected.num_absorbed, result.num_absorbed);
        }
    }

  protected:
    CollectResults collect_;
    SurfaceTestResults results_;
};

//---------------------------------------------------------------------------//
class SurfacePhysicsIntegrationBackscatterTest
    : public SurfacePhysicsIntegrationTest
{
  public:
    void setup_surface_models(inp::SurfacePhysics& input) const final
    {
        PhysSurfaceId phys_surface{0};

        // center-top surface

        input.materials.push_back({});
        input.roughness.polished.emplace(phys_surface, inp::NoRoughness{});
        input.reflectivity.fresnel.emplace(phys_surface,
                                           inp::FresnelReflection{});

        // Only back-scattering

        input.interaction.trivial.emplace(phys_surface,
                                          TrivialInteractionMode::backscatter);
    }
};

//---------------------------------------------------------------------------//
class SurfacePhysicsIntegrationAbsorbTest : public SurfacePhysicsIntegrationTest
{
  public:
    void setup_surface_models(inp::SurfacePhysics& input) const final
    {
        PhysSurfaceId phys_surface{0};

        // center-top surface

        input.materials.push_back({});
        input.roughness.polished.emplace(phys_surface, inp::NoRoughness{});
        input.reflectivity.fresnel.emplace(phys_surface,
                                           inp::FresnelReflection{});

        // Only absorption

        input.interaction.trivial.emplace(phys_surface,
                                          TrivialInteractionMode::absorb);
    }
};

//---------------------------------------------------------------------------//
class SurfacePhysicsIntegrationTransmitTest
    : public SurfacePhysicsIntegrationTest
{
  public:
    void setup_surface_models(inp::SurfacePhysics& input) const final
    {
        PhysSurfaceId phys_surface{0};

        // center-top surface

        input.materials.push_back({});
        input.roughness.polished.emplace(phys_surface, inp::NoRoughness{});
        input.reflectivity.fresnel.emplace(phys_surface,
                                           inp::FresnelReflection{});

        // Only transmission

        input.interaction.trivial.emplace(phys_surface,
                                          TrivialInteractionMode::transmit);
    }
};

//---------------------------------------------------------------------------//
class SurfacePhysicsIntegrationFresnelTest
    : public SurfacePhysicsIntegrationTest
{
  public:
    void setup_surface_models(inp::SurfacePhysics& input) const final
    {
        PhysSurfaceId phys_surface{0};

        // center-top surface

        input.materials.push_back({});
        input.roughness.polished.emplace(phys_surface, inp::NoRoughness{});
        input.reflectivity.fresnel.emplace(phys_surface,
                                           inp::FresnelReflection{});

        // Fresnel refraction / reflection

        input.interaction.dielectric.emplace(
            phys_surface,
            inp::DielectricInteraction::from_dielectric(
                inp::ReflectionForm::from_spike()));
    }
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
// Only back-scattering
TEST_F(SurfacePhysicsIntegrationBackscatterTest, backscatter)
{
    std::vector<real_type> angles{0, 30, 60};

    SurfaceTestResults expected;
    expected.num_reflected = {100, 100, 100};
    expected.num_refracted = {0, 0, 0};
    expected.num_absorbed = {0, 0, 0};

    this->reference_run(angles, expected);
}

//---------------------------------------------------------------------------//
// Only absorption
TEST_F(SurfacePhysicsIntegrationAbsorbTest, absorb)
{
    std::vector<real_type> angles{0, 30, 60};

    SurfaceTestResults expected;
    expected.num_refracted = {0, 0, 0};
    expected.num_reflected = {0, 0, 0};
    expected.num_absorbed = {100, 100, 100};

    this->reference_run(angles, expected);
}

//---------------------------------------------------------------------------//
// Only transmission
TEST_F(SurfacePhysicsIntegrationTransmitTest, transmit)
{
    std::vector<real_type> angles{0, 30, 60};

    SurfaceTestResults expected;
    expected.num_refracted = {100, 100, 100};
    expected.num_reflected = {0, 0, 0};
    expected.num_absorbed = {0, 0, 0};

    this->reference_run(angles, expected);
}

//---------------------------------------------------------------------------//
// Fresnel reflection / refraction
TEST_F(SurfacePhysicsIntegrationFresnelTest, fresnel)
{
    std::vector<real_type> angles{
        0,
        10,
        20,
        30,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        60,
        70,
        80,
    };

    SurfaceTestResults expected;
    expected.num_absorbed = {
        0u,
        0u,
        0u,
        0u,
        0u,
        0u,
        0u,
        0u,
        0u,
        0u,
        0u,
        0u,
        0u,
        0u,
        0u,
        0u,
        0u,
        0u,
    };
    expected.num_reflected = {
        2u,
        0u,
        3u,
        4u,
        15u,
        11u,
        9u,
        17u,
        18u,
        34u,
        27u,
        42u,
        60u,
        100u,
        100u,
        100u,
        100u,
        100u,
    };
    expected.num_refracted = {
        98u,
        100u,
        97u,
        96u,
        85u,
        89u,
        91u,
        83u,
        82u,
        66u,
        73u,
        58u,
        40u,
        0u,
        0u,
        0u,
        0u,
        0u,
    };

    this->reference_run(angles, expected);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
