//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/Generator.test.cc
//---------------------------------------------------------------------------//
#include <set>
#include <utility>
#include <vector>

#include "corecel/Types.hh"
#include "corecel/random/distribution/PoissonDistribution.hh"
#include "geocel/UnitUtils.hh"
#include "celeritas/inp/StandaloneInput.hh"
#include "celeritas/optical/Runner.hh"
#include "celeritas/optical/gen/GeneratorData.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{

// Reference results:
// - Double precision
// - Not vecgeom surface
constexpr bool reference_configuration
    = ((CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
       && !CELERITAS_VECGEOM_SURFACE
       && CELERITAS_CORE_RNG == CELERITAS_CORE_RNG_XORWOW);

//---------------------------------------------------------------------------//
// TEST FIXTURES
//---------------------------------------------------------------------------//

class LArSphereGeneratorTest : public Test
{
  public:
    using VecDistribution = std::vector<optical::GeneratorDistributionData>;

  public:
    void SetUp() override
    {
        // Set geometry filename
        osi_.problem.model.geometry
            = Test::test_data_path("geocel", "lar-sphere.gdml");

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

    //! Buffer host distribution data for Cherenkov and scintillation
    VecDistribution make_distributions(size_type count)
    {
        using GT = GeneratorType;

        std::vector<GT> types{GT::cherenkov, GT::scintillation};
        std::mt19937 rng;

        PoissonDistribution<real_type> sample_num_photons(100);

        optical::GeneratorDistributionData data;
        data.step_length = from_cm(0.2);
        data.charge = units::ElementaryCharge{-1};
        data.material = OptMatId(0);
        data.continuous_edep_fraction = 1;
        data.points[StepPoint::pre]
            = {units::LightSpeed(0.7), 0, from_cm(Real3{0, 0, 0})};
        data.points[StepPoint::post]
            = {units::LightSpeed(0.6), 1e-11, from_cm(Real3{0, 0, 0.2})};

        VecDistribution result(count, data);
        for (auto i : range(count))
        {
            result[i].type = types[i % types.size()];
            result[i].num_photons = sample_num_photons(rng);
            CELER_ASSERT(result[i]);
        }
        return result;
    }

  protected:
    inp::OpticalStandaloneInput osi_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(LArSphereGeneratorTest, primary)
{
    // Create primary generator input
    osi_.problem.generator = [] {
        inp::OpticalPrimaryGenerator gen;
        gen.primaries = 65536;
        gen.energy = inp::MonoenergeticDistribution{1e-5};
        gen.angle = inp::IsotropicDistribution{};
        gen.shape = inp::PointDistribution{{0, 0, 0}};
        return gen;
    }();

    // Set number of track slots
    osi_.problem.capacity.tracks = 16384;

    // Construct the runner and transport optical primaries
    auto result = optical::Runner(std::move(osi_))();

    if (reference_configuration)
    {
        EXPECT_EQ(68884, result.counters.steps);
        EXPECT_EQ(6, result.counters.step_iters);
    }
    EXPECT_EQ(1, result.counters.flushes);
    ASSERT_EQ(1, result.counters.generators.size());

    auto const& gen = result.counters.generators.front();
    EXPECT_EQ(0, gen.buffer_size);
    EXPECT_EQ(0, gen.num_pending);
    EXPECT_EQ(65536, gen.num_generated);
}

TEST_F(LArSphereGeneratorTest, direct)
{
    osi_.problem.generator = inp::OpticalDirectGenerator{};
    osi_.problem.capacity.tracks = 32;

    // Create direct generator action
    std::vector<optical::TrackInitializer> const inits(
        128,
        optical::TrackInitializer{units::MevEnergy{1e-5},
                                  Real3{0, 0, 0},
                                  Real3{1, 0, 0},
                                  Real3{0, 1, 0},
                                  0,
                                  {},  // primary
                                  ImplVolumeId{0}});

    // Construct the runner and transport optical primaries
    auto result = optical::Runner(std::move(osi_))(make_span(inits));

    if (reference_configuration
        && CELERITAS_CORE_GEO != CELERITAS_CORE_GEO_GEANT4)
    {
        EXPECT_EQ(135, result.counters.steps);
        EXPECT_EQ(6, result.counters.step_iters);
    }
    EXPECT_EQ(1, result.counters.flushes);
    ASSERT_EQ(1, result.counters.generators.size());

    auto const& gen = result.counters.generators.front();
    EXPECT_EQ(128, gen.buffer_size);
    EXPECT_EQ(0, gen.num_pending);
    EXPECT_EQ(128, gen.num_generated);
}

TEST_F(LArSphereGeneratorTest, offload)
{
    // Generate Cherenkov and scintillation photons
    osi_.problem.generator = inp::OpticalOffloadGenerator{};

    // Enable Cherenkov and scintillation
    osi_.geant_setup.cherenkov.enable = true;
    osi_.geant_setup.scintillation.enable = true;

    // Set number of track slots and number of distributions
    osi_.problem.capacity.tracks = 4096;
    osi_.problem.capacity.generators = 512;

    // Enable action times
    osi_.problem.timers.action = true;

    // Create host distributions and copy to generator
    auto const host_data
        = this->make_distributions(osi_.problem.capacity.generators);

    // Construct the runner and transport optical primaries
    auto result = optical::Runner(std::move(osi_))(make_span(host_data));

    EXPECT_EQ(1, result.counters.flushes);
    ASSERT_EQ(1, result.counters.generators.size());

    auto const& gen = result.counters.generators.front();
    EXPECT_EQ(512, gen.buffer_size);
    EXPECT_EQ(0, gen.num_pending);

    if (reference_configuration)
    {
        EXPECT_EQ(51226, gen.num_generated);
        EXPECT_EQ(53460, result.counters.steps);
        EXPECT_EQ(15, result.counters.step_iters);
    }

    // Check accumulated action times
    std::set<std::string> labels;
    for (auto const& [label, time] : result.action_times)
    {
        labels.insert(label);
        EXPECT_GT(time, 0);
    }
    static std::string const expected_labels[] = {
        "absorption",
        "along-step",
        "locate-vacancies",
        "optical-boundary-init",
        "optical-boundary-post",
        "optical-discrete-select",
        "optical-generate",
        "optical-surface-stepping",
        "pre-step",
        "tracking-cut",
    };
    EXPECT_VEC_EQ(expected_labels, labels);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
