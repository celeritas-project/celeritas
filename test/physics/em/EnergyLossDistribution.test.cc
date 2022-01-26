//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file EnergyLossDistribution.test.cc
//---------------------------------------------------------------------------//
#include "physics/em/EnergyLossDistribution.hh"

#include "base/CollectionStateStore.hh"
#include "random/DiagnosticRngEngine.hh"
#include "physics/base/CutoffParams.hh"
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
    using HostValue = FluctuationData<Ownership::value, MemSpace::host>;
    using HostRef = FluctuationData<Ownership::const_reference, MemSpace::host>;
    using MaterialStateStore
        = CollectionStateStore<MaterialStateData, MemSpace::host>;
    using ParticleStateStore
        = CollectionStateStore<ParticleStateData, MemSpace::host>;

    void SetUp() override
    {
        using namespace celeritas::constants;
        using namespace celeritas::units;
        namespace pdg = celeritas::pdg;

        constexpr auto stable
            = celeritas::ParticleRecord::stable_decay_constant();

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

        // Construct shared cutoff params
        CutoffParams::Input cut_inp{
            particles, materials, {{pdg::electron(), {{MevEnergy{1e-3}, 0}}}}};
        cutoffs = std::make_shared<CutoffParams>(std::move(cut_inp));

        // Construct states for a single host thread
        particle_state = ParticleStateStore(*particles, 1);
        material_state = MaterialStateStore(*materials, 1);

        // Construct energy loss fluctuation model parameters
        fluct.electron_id   = particles->find(pdg::electron());
        fluct.electron_mass = particles->get(fluct.electron_id).mass().value();
        FluctuationParameters params;
        params.oscillator_strength = {8. / 9, 1. / 9};
        params.binding_energy      = {1.317071809191344e-4, 3.24e-3};
        params.log_binding_energy  = {std::log(params.binding_energy[0]),
                                     std::log(params.binding_energy[1])};
        make_builder(&fluct.params).push_back(params);
        fluct_ref = fluct;
    }

    HostValue                         fluct;
    HostRef                           fluct_ref;
    std::shared_ptr<MaterialParams>   materials;
    std::shared_ptr<ParticleParams>   particles;
    std::shared_ptr<CutoffParams>     cutoffs;
    ParticleStateStore                particle_state;
    MaterialStateStore                material_state;
    DiagnosticRngEngine<std::mt19937> rng;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(EnergyLossDistributionTest, gaussian)
{
    ParticleTrackView particle(
        particles->host_ref(), particle_state.ref(), ThreadId{0});
    particle = {ParticleId{1}, MevEnergy{1e-2}};
    MaterialTrackView material(
        materials->host_ref(), material_state.ref(), ThreadId{0});
    material = {MaterialId{0}};
    CutoffView cutoff(cutoffs->host_ref(), MaterialId{0});
    MevEnergy  mean_loss{0.1};

    int                 num_samples = 5000;
    std::vector<double> mean;
    std::vector<double> counts(20);
    double              upper = 7.0;
    double              lower = 0.0;
    double              width = (upper - lower) / counts.size();

    // Larger step samples from gamma distribution, smaller step from Gaussian
    for (double step : {5e-2, 5e-4})
    {
        double                 sum = 0;
        EnergyLossDistribution sample_loss(
            fluct_ref, cutoff, material, particle, mean_loss, step);
        for (CELER_MAYBE_UNUSED int i : celeritas::range(num_samples))
        {
            auto loss = sample_loss(rng).value();
            auto bin  = size_type((loss - lower) / width);
            CELER_ASSERT(bin < counts.size());
            counts[bin]++;
            sum += loss;
        }
        mean.push_back(sum / num_samples);
    }
    const double expected_counts[] = {9646, 150, 87, 35, 24, 21, 13, 9, 6, 1,
                                      1,    2,   2,  1,  1,  0,  0,  1, 0, 0};
    const double expected_mean[]   = {0.0952213970906181, 0.1008228960123};
    EXPECT_VEC_SOFT_EQ(expected_counts, counts);
    EXPECT_VEC_SOFT_EQ(expected_mean, mean);
    EXPECT_EQ(60410, rng.count());
}

TEST_F(EnergyLossDistributionTest, urban)
{
    ParticleTrackView particle(
        particles->host_ref(), particle_state.ref(), ThreadId{0});
    particle = {ParticleId{0}, MevEnergy{100}};
    MaterialTrackView material(
        materials->host_ref(), material_state.ref(), ThreadId{0});
    material = {MaterialId{0}};
    CutoffView cutoff(cutoffs->host_ref(), MaterialId{0});
    MevEnergy  mean_loss{0.01};
    double     step = 0.01;

    int                 num_samples = 10000;
    std::vector<double> counts(20);
    double              upper = 0.03;
    double              lower = 0.0;
    double              width = (upper - lower) / counts.size();
    double              sum   = 0;

    EnergyLossDistribution sample_loss(
        fluct_ref, cutoff, material, particle, mean_loss, step);
    for (CELER_MAYBE_UNUSED int i : celeritas::range(num_samples))
    {
        auto loss = sample_loss(rng).value();
        auto bin  = size_type((loss - lower) / width);
        CELER_ASSERT(bin < counts.size());
        counts[bin]++;
        sum += loss;
    }
    const double expected_counts[] = {0,    0,   15,  223, 1174, 2398, 2656,
                                      1835, 884, 394, 160, 125,  77,   31,
                                      17,   8,   3,   0,   0,    0};
    EXPECT_VEC_SOFT_EQ(expected_counts, counts);
    EXPECT_SOFT_EQ(9.9757788697025472e-3, sum / num_samples);
    EXPECT_EQ(551188, rng.count());
}
