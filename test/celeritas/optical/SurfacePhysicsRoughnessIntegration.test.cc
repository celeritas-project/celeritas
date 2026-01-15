//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/SurfacePhysicsRoughnessIntegration.test.cc
//---------------------------------------------------------------------------//
#include "corecel/math/Turn.hh"
#include "corecel/random/Histogram.hh"

#include "SurfacePhysicsIntegrationTestBase.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace optical
{
namespace test
{
using ::celeritas::test::Histogram;

//---------------------------------------------------------------------------//
/*!
 * Collect results based on the track's direction dot produced with respect to
 * the surface normal.
 *
 * The surface normal is (0,1,0), so the dot product is just the y-component.
 * This gives a distribution of reflected and refracted angles.
 */
struct CollectResults
{
    Histogram reflection_cosine{20, {-1, 1}};
    size_type num_failed{0};

    //! Score track
    void operator()(CoreTrackView const& track)
    {
        if (track.sim().status() == TrackStatus::alive)
        {
            reflection_cosine(track.geometry().dir()[1]);
            return;
        }
        num_failed++;
    }
};

//---------------------------------------------------------------------------//
// TEST CHASSIS
//---------------------------------------------------------------------------//
/*!
 * Base class for testing surface roughness models.
 *
 * Sub-classes should use Fresnel reflection with the lobe mode to ensure the
 * local facet normal is used for reflection.
 */
class SurfacePhysicsRoughnessIntegrationTest
    : public SurfacePhysicsIntegrationTestBase
{
  public:
    /*!
     * Run for a certain number of iterations and compare to the expected
     * distribution.
     */
    void run(size_type loops, std::vector<size_type> const& expected)
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
            EXPECT_VEC_EQ(expected, collect_.reflection_cosine.counts());
        }
    }

  protected:
    CollectResults collect_;
};

//---------------------------------------------------------------------------//
/*!
 * Polished roughness model.
 */
class SurfacePhysicsIntegrationPolishedTest
    : public SurfacePhysicsRoughnessIntegrationTest
{
  public:
    void setup_surface_models(inp::SurfacePhysics& input) const final
    {
        PhysSurfaceId phys_surface{0};

        // center-top surface

        input.materials.push_back({});
        input.reflectivity.fresnel.emplace(phys_surface,
                                           inp::FresnelReflection{});
        input.interaction.dielectric.emplace(
            phys_surface,
            inp::DielectricInteraction::from_dielectric(
                inp::ReflectionForm::from_lobe()));

        // polished roughness

        input.roughness.polished.emplace(phys_surface, inp::NoRoughness{});
    }
};

//---------------------------------------------------------------------------//
/*!
 * Uniform smear roughness model.
 */
class SurfacePhysicsIntegrationSmearTest
    : public SurfacePhysicsRoughnessIntegrationTest
{
  public:
    void setup_surface_models(inp::SurfacePhysics& input) const final
    {
        PhysSurfaceId phys_surface{0};

        // center-top surface

        input.materials.push_back({});
        input.reflectivity.fresnel.emplace(phys_surface,
                                           inp::FresnelReflection{});
        input.interaction.dielectric.emplace(
            phys_surface,
            inp::DielectricInteraction::from_dielectric(
                inp::ReflectionForm::from_lobe()));

        // smear roughness

        input.roughness.smear.emplace(phys_surface, inp::SmearRoughness{0.8});
    }
};

//---------------------------------------------------------------------------//
/*!
 * Gaussian roughness model.
 */
class SurfacePhysicsIntegrationGaussianTest
    : public SurfacePhysicsRoughnessIntegrationTest
{
  public:
    void setup_surface_models(inp::SurfacePhysics& input) const final
    {
        PhysSurfaceId phys_surface{0};

        // center-top surface

        input.materials.push_back({});
        input.reflectivity.fresnel.emplace(phys_surface,
                                           inp::FresnelReflection{});
        input.interaction.dielectric.emplace(
            phys_surface,
            inp::DielectricInteraction::from_dielectric(
                inp::ReflectionForm::from_lobe()));

        // Gaussian roughness

        input.roughness.gaussian.emplace(phys_surface,
                                         inp::GaussianRoughness{0.6});
    }
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
// Only polished
TEST_F(SurfacePhysicsIntegrationPolishedTest, polished)
{
    std::vector<size_type> expected{
        15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 985,
    };

    this->run(10, expected);
}

//---------------------------------------------------------------------------//
// Only uniform smear
TEST_F(SurfacePhysicsIntegrationSmearTest, smear)
{
    std::vector<size_type> expected{
        4, 11, 6, 5, 7, 4, 3, 4, 7, 15, 0, 0, 0, 1, 0, 0, 0, 1, 34, 898,
    };

    this->run(10, expected);
}

//---------------------------------------------------------------------------//
// Only Gaussian roughness
TEST_F(SurfacePhysicsIntegrationGaussianTest, gaussian)
{
    std::vector<size_type> expected{
        4,  17, 14, 23, 20, 27, 26, 21, 36, 33,
        22, 11, 21, 9,  11, 13, 13, 14, 57, 608,
    };

    this->run(10, expected);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
