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

    EnergyLossDistribution sample_loss(
        material, particle, cutoff_energy, mean_loss, step);
    for (CELER_MAYBE_UNUSED int i : celeritas::range(num_samples))
    {
        auto bin = size_type((sample_loss(rng).value() - lower) / width);
        CELER_ASSERT(bin < counts.size());
        counts[bin]++;
    }
    const double expected_counts[] = {0,    0,   13,  232, 1155, 2359, 2628,
                                      1866, 910, 378, 196, 129,  81,   36,
                                      7,    9,   1,   0,   0,    0};
    EXPECT_VEC_SOFT_EQ(expected_counts, counts);
    EXPECT_EQ(554606, rng.count());
}

TEST_F(EnergyLossDistributionTest, tmp)
{
    int num_samples = 100;

    ParticleTrackView particle(
        particles->host_pointers(), particle_state.ref(), ThreadId{0});
    particle = {ParticleId{0}, MevEnergy{100}};
    MaterialView material(materials->host_pointers(), MaterialId{0});
    MevEnergy    cutoff_energy{0.001};
    MevEnergy    mean_loss{0.01};
    double       step = 0.01;

    std::vector<double>    energy_loss;
    EnergyLossDistribution sample_loss(
        material, particle, cutoff_energy, mean_loss, step);
    for (CELER_MAYBE_UNUSED int i : celeritas::range(num_samples))
    {
        energy_loss.push_back(sample_loss(rng).value());
    }
    // PRINT_EXPECTED(energy_loss);
    const double expected_energy_loss[]
        = {0.01184309261896,  0.009989128372117, 0.008946028482641,
           0.01484290044747,  0.00822117079274,  0.008109235273801,
           0.00893236115937,  0.009672580968466, 0.008037976376555,
           0.009738498985771, 0.005778947923171, 0.007654843958958,
           0.01453755779158,  0.009825620427781, 0.006276460711482,
           0.01294048905283,  0.007855153618489, 0.01038059175871,
           0.006183201871706, 0.0113492831039,   0.01050672239041,
           0.007315270660573, 0.01171160037848,  0.005971815925004,
           0.006902136558448, 0.01062480663845,  0.008989188095438,
           0.007648303660447, 0.009129010370036, 0.009795584845351,
           0.009777140951788, 0.009891549245388, 0.005210763866184,
           0.01208525550297,  0.01098607921519,  0.009305248701818,
           0.00679828468798,  0.01304072569044,  0.01228038586111,
           0.01213704233134,  0.005441127681669, 0.01242045446784,
           0.01256597093612,  0.009381776743696, 0.0116954012394,
           0.009975787799318, 0.0125560373213,   0.01105049603704,
           0.00874079829816,  0.009675229543464, 0.009030846408679,
           0.01036345683773,  0.01086855971005,  0.01250725066557,
           0.009147458009162, 0.01338775322564,  0.008222901765401,
           0.009547355621117, 0.01138431598518,  0.008363476344941,
           0.008648045336147, 0.008307996915043, 0.009423514740862,
           0.01092445342598,  0.01260284088479,  0.007784420645361,
           0.009055459107526, 0.0119152798911,   0.008983407986397,
           0.009414574317872, 0.01093985562104,  0.00857497177434,
           0.008764548761478, 0.01142247521669,  0.01155306059894,
           0.005809515569757, 0.01305141102623,  0.01217468336714,
           0.009476656518167, 0.007827303419397, 0.008806925245409,
           0.008991590780306, 0.009885197313094, 0.008176305056976,
           0.00886059015207,  0.01217411226643,  0.007477862204882,
           0.01226738848954,  0.008988871512762, 0.01070006227612,
           0.01432502509508,  0.006485588359661, 0.009540387332741,
           0.008352205390102, 0.009843645532985, 0.01206555487664,
           0.01075361009745,  0.01092888105009,  0.01091430836723,
           0.009401260664985};
    EXPECT_VEC_SOFT_EQ(expected_energy_loss, energy_loss);
    EXPECT_EQ(5550, rng.count());
}
