//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/SurfacePhysicsInteractionIntegration.test.cc
//---------------------------------------------------------------------------//

#include "corecel/math/Turn.hh"

#include "SurfacePhysicsIntegrationTestBase.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace optical
{
namespace test
{
constexpr Turn degree{real_type{1} / 360};
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

class SurfacePhysicsInteractionIntegrationTest
    : public SurfacePhysicsIntegrationTestBase
{
  public:
    SurfaceTestResults run(std::vector<RealTurn> const& angles)
    {
        this->create_collector<CollectResults>(collect_);

        this->initialize_run();

        // Run over angles
        SurfaceTestResults results;
        for (auto angle : angles)
        {
            collect_.reset();

            this->run_step(angle);

            EXPECT_EQ(0, collect_.num_failed);
            results.num_absorbed.push_back(collect_.num_absorbed);
            results.num_reflected.push_back(collect_.num_reflected);
            results.num_refracted.push_back(collect_.num_refracted);
        }

        return results;
    }

    void reference_run(std::vector<RealTurn> const& angles,
                       SurfaceTestResults const& expected)
    {
        auto result = this->run(angles);
        if (reference_configuration)
        {
            EXPECT_EQ(expected.num_reflected, result.num_reflected);
            EXPECT_EQ(expected.num_refracted, result.num_refracted);
            EXPECT_EQ(expected.num_absorbed, result.num_absorbed);
        }
    }

  protected:
    CollectResults collect_;
};

//---------------------------------------------------------------------------//
class SurfacePhysicsIntegrationBackscatterTest
    : public SurfacePhysicsInteractionIntegrationTest
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
class SurfacePhysicsIntegrationAbsorbTest
    : public SurfacePhysicsInteractionIntegrationTest
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
    : public SurfacePhysicsInteractionIntegrationTest
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
    : public SurfacePhysicsInteractionIntegrationTest
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
class SurfacePhysicsIntegrationOnlyReflectionPolishedTest
    : public SurfacePhysicsInteractionIntegrationTest
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

        // Only polished (specular spike) reflection

        input.interaction.only_reflection.emplace(
            phys_surface, ReflectionMode::specular_spike);
    }
};

//---------------------------------------------------------------------------//
class SurfacePhysicsIntegrationOnlyReflectionGroundTest
    : public SurfacePhysicsInteractionIntegrationTest
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

        // Only ground (diffuse lobe) reflection

        input.interaction.only_reflection.emplace(
            phys_surface, ReflectionMode::diffuse_lobe);
    }
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
// Only back-scattering
TEST_F(SurfacePhysicsIntegrationBackscatterTest, backscatter)
{
    std::vector<RealTurn> angles{RealTurn{0}, 30 * degree, 60 * degree};

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
    std::vector<RealTurn> angles{RealTurn{0}, 30 * degree, 60 * degree};

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
    std::vector<RealTurn> angles{RealTurn{0}, 30 * degree, 60 * degree};

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
    std::vector<RealTurn> angles{
        RealTurn{0},
        10 * degree,
        20 * degree,
        30 * degree,
        40 * degree,
        41 * degree,
        42 * degree,
        43 * degree,
        44 * degree,
        45 * degree,
        46 * degree,
        47 * degree,
        48 * degree,
        49 * degree,
        50 * degree,
        60 * degree,
        70 * degree,
        80 * degree,
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
// Only polished reflection
TEST_F(SurfacePhysicsIntegrationOnlyReflectionPolishedTest, polished)
{
    std::vector<RealTurn> angles{RealTurn{0}, 30 * degree, 60 * degree};

    SurfaceTestResults expected;
    expected.num_refracted = {0, 0, 0};
    expected.num_reflected = {100, 100, 100};
    expected.num_absorbed = {0, 0, 0};

    this->reference_run(angles, expected);
}

//---------------------------------------------------------------------------//
// Only ground reflection
TEST_F(SurfacePhysicsIntegrationOnlyReflectionGroundTest, ground)
{
    std::vector<RealTurn> angles{RealTurn{0}, 30 * degree, 60 * degree};

    SurfaceTestResults expected;
    expected.num_refracted = {0, 0, 0};
    expected.num_reflected = {100, 100, 100};
    expected.num_absorbed = {0, 0, 0};

    this->reference_run(angles, expected);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
