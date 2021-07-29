//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file EnergyLossDistribution.test.cc
//---------------------------------------------------------------------------//
#include "physics/base/EnergyLossDistribution.hh"

#include "base/CollectionStateStore.hh"
#include "random/DiagnosticRngEngine.hh"
#include "physics/base/ParticleParams.hh"
#include "physics/material/MaterialParams.hh"
#include "celeritas_test.hh"

using namespace celeritas;
using namespace celeritas_test;
using celeritas::units::MevEnergy;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class EnergyLossDistributionTest : public celeritas::Test
{
  protected:
    using ParticleStateStore
        = CollectionStateStore<ParticleStateData, MemSpace::host>;

    void SetUp() override
    {
        using namespace celeritas::constants;
        using namespace celeritas::units;
        namespace pdg = celeritas::pdg;

        constexpr auto stable = celeritas::ParticleDef::stable_decay_constant();

        // Set up shared material data
        MaterialParams::Input mat_inp;
        mat_inp.elements  = {{18, AmuMass{39.948}, "Ar"}};
        mat_inp.materials = {
            {1.0 * na_avogadro,
             293.0,
             MatterState::solid,
             {{ElementId{0}, 1.0}},
             "Ar"},
        };
        materials = std::make_shared<MaterialParams>(std::move(mat_inp));

        // Set up shared particle data
        ParticleParams::Input par_inp{{"electron",
                                       pdg::electron(),
                                       MevMass{0.5109989461},
                                       ElementaryCharge{-1},
                                       stable},
                                      {"mu_minus",
                                       pdg::mu_minus(),
                                       MevMass{105.6583745},
                                       ElementaryCharge{-1},
                                       stable}};
        particles = std::make_shared<ParticleParams>(std::move(par_inp));

        // Construct particle state for a single host thread
        particle_state = ParticleStateStore(*particles, 1);
    }

    std::shared_ptr<MaterialParams>   materials;
    std::shared_ptr<ParticleParams>   particles;
    ParticleStateStore                particle_state;
    DiagnosticRngEngine<std::mt19937> rng;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(EnergyLossDistributionTest, gaussian)
{
    ParticleTrackView particle(
        particles->host_pointers(), particle_state.ref(), ThreadId{0});
    particle = {ParticleId{1}, MevEnergy{1e-2}};
    MaterialView material(materials->host_pointers(), MaterialId{0});
    MevEnergy    cutoff_energy{0.001};
    MevEnergy    mean_loss{0.1};

    int                 num_samples = 5000;
    std::vector<double> counts(20);
    double              upper = 0.3;
    double              lower = 0.0;
    double              width = (upper - lower) / counts.size();

    // Larger step samples from gamma distribution, smaller step from Gaussian
    for (double step : {1e-2, 1e-4})
    {
        EnergyLossDistribution sample_loss(
            material, particle, cutoff_energy, mean_loss, step);
        for (CELER_MAYBE_UNUSED int i : celeritas::range(num_samples))
        {
            auto bin = size_type((sample_loss(rng).value() - lower) / width);
            CELER_ASSERT(bin < counts.size());
            counts[bin]++;
        }
    }
    const double expected_counts[] = {4652, 244, 287, 381, 571, 762, 826,
                                      744,  631, 448, 261, 121, 54,  15,
                                      0,    1,   1,   0,   0,   1};
    EXPECT_VEC_SOFT_EQ(expected_counts, counts);
    EXPECT_EQ(60410, rng.count());
}

TEST_F(EnergyLossDistributionTest, urban)
{
    ParticleTrackView particle(
        particles->host_pointers(), particle_state.ref(), ThreadId{0});
    particle = {ParticleId{0}, MevEnergy{100}};
    MaterialView material(materials->host_pointers(), MaterialId{0});
    MevEnergy    cutoff_energy{0.001};
    MevEnergy    mean_loss{0.01};
    double       step = 0.01;

    int                 num_samples = 10000;
    std::vector<double> counts(20);
    double              upper = 0.03;
    double              lower = 0.0;
    double              width = (upper - lower) / counts.size();
    double              sum   = 0;

    EnergyLossDistribution sample_loss(
        material, particle, cutoff_energy, mean_loss, step);
    for (CELER_MAYBE_UNUSED int i : celeritas::range(num_samples))
    {
        auto loss = sample_loss(rng).value();
        auto bin  = size_type((loss - lower) / width);
        CELER_ASSERT(bin < counts.size());
        counts[bin]++;
        sum += loss;
    }
    const double expected_counts[] = {0,    0,   13,  232, 1155, 2359, 2628,
                                      1866, 910, 378, 196, 129,  81,   36,
                                      7,    9,   1,   0,   0,    0};
    EXPECT_VEC_SOFT_EQ(expected_counts, counts);
    EXPECT_SOFT_NEAR(mean_loss.value(), sum / num_samples, 1e-3);
    EXPECT_EQ(554606, rng.count());
}
