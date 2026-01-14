//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/Runner.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/optical/Runner.hh"

#include <utility>

#include "corecel/Types.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// TEST FIXTURES
//---------------------------------------------------------------------------//

class LArSphereRunnerTest : public Test
{
  public:
    void SetUp() override
    {
        // TODO: How should we disable the device for tests? (This won't work
        // because the device is activated in test_main)
        osi_.system.environment["CELER_DISABLE_DEVICE"] = "1";

        // Set geometry filename
        osi_.problem.model.geometry
            = Test::test_data_path("geocel", "lar-sphere.gdml");

        // Create primary generator input
        osi_.problem.generator = [] {
            inp::OpticalPrimaryGenerator gen;
            gen.primaries = 65536;
            gen.energy = inp::MonoenergeticDistribution{1e-5};
            gen.angle = inp::IsotropicDistribution{};
            gen.shape = inp::PointDistribution{{0, 0, 0}};
            return gen;
        }();

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
            auto opt = GeantOpticalPhysicsOptions::deactivated();
            opt.absorption = true;
            return opt;
        }();
    }

  protected:
    inp::OpticalStandaloneInput osi_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(LArSphereRunnerTest, primary_generator)
{
    // Construct the runner and transport optical primaries
    auto result = optical::Runner(std::move(osi_))();

    if ((CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
        && !CELERITAS_VECGEOM_SURFACE)
    {
        EXPECT_EQ(68870, result.counters.steps);
        EXPECT_EQ(18, result.counters.step_iters);
    }
    EXPECT_EQ(1, result.counters.flushes);
    ASSERT_EQ(1, result.counters.generators.size());

    auto const& gen = result.counters.generators.front();
    EXPECT_EQ(0, gen.buffer_size);
    EXPECT_EQ(0, gen.num_pending);
    EXPECT_EQ(65536, gen.num_generated);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
