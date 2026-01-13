//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/SurfacePhysicsRoughnessIntegration.test.cc
//---------------------------------------------------------------------------//
#include <memory>

#include "corecel/random/Histogram.hh"

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
            reflection_cosine(track.geometry().dir()[2]);
            return;
        }
        num_failed++;
    }
};

//---------------------------------------------------------------------------//
/*!
 */
class SurfacePhysicsRoughnessIntegrationTest
    : public SurfacePhysicsIntegrationTestBase
{
  public:
    void run(size_type loops, std::vector<size_type> const& expected)
    {
        if (reference_configuration)
        {
            // Create collector
            auto& reg = *this->optical_params()->action_reg();
            auto collector
                = std::make_shared<CollectResultsAction<CollectResults>>(
                    reg.next_id(), collect_);
            reg.insert(collector);

            this->initialize_run();

            for ([[maybe_unused]] auto i : range(loops))
            {
                this->run_step(0);
            }

            EXPECT_EQ(0, collect_.num_failed);
            EXPECT_VEC_EQ(expected, collect_.reflection_cosine.counts());

            PRINT_EXPECTED(collect_.reflection_cosine.counts());
        }
    }

  protected:
    CollectResults collect_{};
};

//---------------------------------------------------------------------------//
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

TEST_F(SurfacePhysicsIntegrationPolishedTest, polished)
{
    std::vector<size_type> expected{0};

    this->run(10, expected);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
