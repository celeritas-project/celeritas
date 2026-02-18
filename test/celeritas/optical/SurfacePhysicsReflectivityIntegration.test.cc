//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/SurfacePhysicsReflectivityIntegration.test.cc
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
//---------------------------------------------------------------------------//

struct ReflectivityResults
{
    size_type num_absorbed{0};
    size_type num_transmitted{0};
    size_type num_interacted{0};
};

//---------------------------------------------------------------------------//
/*!
 * Collect results based on whether interacted (back-scattering), transmitted,
 * or absorbed.
 */
struct CollectResults
{
    ReflectivityResults results;
    size_type num_failed{0};

    //! Score track
    void operator()(CoreTrackView const& track)
    {
        if (track.sim().status() == TrackStatus::alive)
        {
            auto vol = track.geometry().volume_instance_id();
            if (vol == VolumeInstanceId{1})
            {
                results.num_interacted++;
                return;
            }
            else if (vol == VolumeInstanceId{2})
            {
                results.num_transmitted++;
                return;
            }
        }
        else if (track.sim().status() == TrackStatus::killed)
        {
            results.num_absorbed++;
            return;
        }

        num_failed++;
    }
};

//---------------------------------------------------------------------------//
// TEST CHASSIS
//---------------------------------------------------------------------------//
/*!
 * Base class for testing surface reflectivity models.
 *
 * Sub-classes should use polished roughness and trivial back-scattering to
 * ensure interaction results correspond to reflections, transmitted results
 * correspond to refractions, and absorb results are the only killed tracks.
 */
class SurfacePhysicsReflectivityIntegrationTest
    : public SurfacePhysicsIntegrationTestBase
{
  public:
    void run(size_type loops, ReflectivityResults const& expected)
    {
        this->create_collector<CollectResults>(collect_);

        this->initialize_run();

        for ([[maybe_unused]] auto i : range(loops))
        {
            this->run_step(RealTurn(0.0));  // along x = 0
        }

        if (reference_configuration)
        {
            EXPECT_EQ(0, collect_.num_failed);
            EXPECT_EQ(expected.num_absorbed, collect_.results.num_absorbed);
            EXPECT_EQ(expected.num_interacted, collect_.results.num_interacted);
            EXPECT_EQ(expected.num_transmitted,
                      collect_.results.num_transmitted);
        }
    }

  private:
    CollectResults collect_;
};

//---------------------------------------------------------------------------//
/*!
 * Fresnel reflectivity model.
 *
 * Always interacts.
 */
class SurfacePhysicsIntegrationFresnelTest
    : public SurfacePhysicsReflectivityIntegrationTest
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
        input.interaction.trivial.emplace(phys_surface,
                                          TrivialInteractionMode::backscatter);
    }
};

//---------------------------------------------------------------------------//
/*!
 * Grid reflectivity model.
 *
 * Interact with a user defined grid probability.
 */
class SurfacePhysicsIntegrationGridTest
    : public SurfacePhysicsReflectivityIntegrationTest
{
  public:
    void setup_surface_models(inp::SurfacePhysics& input) const final
    {
        PhysSurfaceId phys_surface{0};

        // center-top surface

        input.materials.push_back({});
        input.roughness.polished.emplace(phys_surface, inp::NoRoughness{});
        input.reflectivity.grid.emplace(phys_surface, [] {
            inp::GridReflection refl;
            std::vector<double> xs{1e-6, 2e-6, 4e-6, 5e-6, 7e-6, 8e-6};
            refl.reflectivity = inp::Grid{xs, {0.0, 0.7, 0.7, 0.75, 0.33, 0.0}};
            refl.transmittance = inp::Grid{xs, {0, 0.1, 0.1, 0.2, 0.1, 0}};
            return refl;
        }());
        input.interaction.trivial.emplace(phys_surface,
                                          TrivialInteractionMode::backscatter);
    }
};

//---------------------------------------------------------------------------//
/*!
 * Grid reflectivity model with quantum efficiency.
 *
 * Interact with a user defined grid probability.
 */
class SurfacePhysicsIntegrationEfficiencyTest
    : public SurfacePhysicsReflectivityIntegrationTest
{
  public:
    void setup_surface_models(inp::SurfacePhysics& input) const final
    {
        PhysSurfaceId phys_surface{0};

        // center-top surface

        input.materials.push_back({});
        input.roughness.polished.emplace(phys_surface, inp::NoRoughness{});
        input.reflectivity.grid.emplace(phys_surface, [] {
            inp::GridReflection refl;
            std::vector<double> xs{1e-6, 2e-6, 4e-6, 5e-6, 7e-6, 8e-6};
            refl.reflectivity = inp::Grid{xs, {0.0, 0.2, 0.2, 0.75, 0.33, 0.0}};
            refl.transmittance = inp::Grid{xs, {0, 0.1, 0.1, 0.2, 0.1, 0}};
            refl.efficiency = inp::Grid{xs, {0, 0.6, 0.6, 0.1, 0, 0}};
            return refl;
        }());
        input.interaction.trivial.emplace(phys_surface,
                                          TrivialInteractionMode::backscatter);
    }
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
// Only Fresnel
TEST_F(SurfacePhysicsIntegrationFresnelTest, fresnel)
{
    ReflectivityResults expected;
    expected.num_absorbed = 0;
    expected.num_transmitted = 0;
    expected.num_interacted = 10000;

    this->run(100, expected);
}

//---------------------------------------------------------------------------//
// Only user grid
TEST_F(SurfacePhysicsIntegrationGridTest, grid)
{
    ReflectivityResults expected;
    expected.num_absorbed = 1917;
    expected.num_transmitted = 1014;
    expected.num_interacted = 7069;

    this->run(100, expected);
}

//---------------------------------------------------------------------------//
// Test quantum efficiency on a surface
TEST_F(SurfacePhysicsIntegrationEfficiencyTest, efficiency)
{
    ReflectivityResults expected;
    expected.num_absorbed = 2942;
    expected.num_transmitted = 5104;
    expected.num_interacted = 1954;

    this->run(100, expected);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
