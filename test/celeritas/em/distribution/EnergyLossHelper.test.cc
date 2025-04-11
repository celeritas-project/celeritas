//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/distribution/EnergyLossHelper.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/em/distribution/EnergyLossHelper.hh"

#include "corecel/data/CollectionStateStore.hh"
#include "corecel/random/DiagnosticRngEngine.hh"
#include "celeritas/MockTestBase.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/em/distribution/EnergyLossDeltaDistribution.hh"
#include "celeritas/em/distribution/EnergyLossGammaDistribution.hh"
#include "celeritas/em/distribution/EnergyLossGaussianDistribution.hh"
#include "celeritas/em/distribution/EnergyLossUrbanDistribution.hh"
#include "celeritas/em/params/FluctuationParams.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/phys/CutoffParams.hh"
#include "celeritas/phys/ParticleParams.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
using units::MevEnergy;
using EnergySq = RealQuantity<UnitProduct<units::Mev, units::Mev>>;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class MockFluctuationTest : public MockTestBase
{
  protected:
    void SetUp() override
    {
        fluct = std::make_shared<FluctuationParams>(*this->particle(),
                                                    *this->material());
    }

    std::shared_ptr<FluctuationParams const> fluct;
};

//---------------------------------------------------------------------------//

class EnergyLossDistributionTest : public Test
{
  protected:
    using HostValue = HostVal<FluctuationData>;
    using HostRef = HostCRef<FluctuationData>;
    using MaterialStateStore
        = CollectionStateStore<MaterialStateData, MemSpace::host>;
    using ParticleStateStore
        = CollectionStateStore<ParticleStateData, MemSpace::host>;
    using EnergySq = RealQuantity<UnitProduct<units::Mev, units::Mev>>;

    void SetUp() override
    {
        using namespace constants;
        using namespace units;

        // Set up shared material data
        MaterialParams::Input mat_inp;
        mat_inp.elements = {{AtomicNumber{18}, AmuMass{39.948}, {}, "Ar"}};
        mat_inp.materials = {
            {native_value_from(MolCcDensity{1.0}),
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
                                       stable_decay_constant},
                                      {"mu_minus",
                                       pdg::mu_minus(),
                                       MevMass{105.6583745},
                                       ElementaryCharge{-1},
                                       stable_decay_constant}};
        particles = std::make_shared<ParticleParams>(std::move(par_inp));

        // Construct shared cutoff params
        CutoffParams::Input cut_inp{
            particles, materials, {{pdg::electron(), {{MevEnergy{1e-3}, 0}}}}};
        cutoffs = std::make_shared<CutoffParams>(std::move(cut_inp));

        // Construct states for a single host thread
        particle_state = ParticleStateStore(particles->host_ref(), 1);
        material_state = MaterialStateStore(materials->host_ref(), 1);

        // Construct energy loss fluctuation model parameters
        fluct = std::make_shared<FluctuationParams>(*particles, *materials);
    }

    std::shared_ptr<MaterialParams> materials;
    std::shared_ptr<ParticleParams> particles;
    std::shared_ptr<CutoffParams> cutoffs;
    std::shared_ptr<FluctuationParams> fluct;

    ParticleStateStore particle_state;
    MaterialStateStore material_state;
    DiagnosticRngEngine<std::mt19937> rng;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(MockFluctuationTest, data)
{
    auto const& urban = fluct->host_ref().urban;

    {
        // Celerogen: Z=1, I=19.2 eV
        auto const& params = urban[MaterialId{0}];
        EXPECT_SOFT_EQ(1, params.oscillator_strength[0]);
        EXPECT_SOFT_EQ(0, params.oscillator_strength[1]);
        EXPECT_SOFT_EQ(19.2e-6, params.binding_energy[0]);
        EXPECT_SOFT_EQ(1e-5, params.binding_energy[1]);
    }
    {
        // Celer composite: Z_eff = 10.3, I=150.7 eV
        auto const& params = urban[MaterialId{2}];
        EXPECT_SOFT_EQ(0.80582524271844658, params.oscillator_strength[0]);
        EXPECT_SOFT_EQ(0.1941747572815534, params.oscillator_strength[1]);
        EXPECT_SOFT_EQ(9.4193231228829647e-5, params.binding_energy[0]);
        EXPECT_SOFT_EQ(1.0609e-3, params.binding_energy[1]);
    }
}

//---------------------------------------------------------------------------//

TEST_F(EnergyLossDistributionTest, none)
{
    ParticleTrackView particle(
        particles->host_ref(), particle_state.ref(), TrackSlotId{0});
    particle = {ParticleId{0}, MevEnergy{1e-2}};
    MaterialTrackView material(
        materials->host_ref(), material_state.ref(), TrackSlotId{0});
    material = {MaterialId{0}};
    CutoffView cutoff(cutoffs->host_ref(), MaterialId{0});
    MevEnergy mean_loss{2e-6};

    // Tiny step, little energy loss
    real_type step = 1e-6 * units::centimeter;
    EnergyLossHelper helper(
        fluct->host_ref(), cutoff, material, particle, mean_loss, step);
    EXPECT_EQ(EnergyLossFluctuationModel::none, helper.model());

    EnergyLossDeltaDistribution sample_loss(helper);
    EXPECT_EQ(mean_loss, sample_loss(rng));
    EXPECT_EQ(0, rng.count());
}

TEST_F(EnergyLossDistributionTest, gaussian)
{
    ParticleTrackView particle(
        particles->host_ref(), particle_state.ref(), TrackSlotId{0});
    particle = {ParticleId{1}, MevEnergy{1e-2}};
    MaterialTrackView material(
        materials->host_ref(), material_state.ref(), TrackSlotId{0});
    material = {MaterialId{0}};
    CutoffView cutoff(cutoffs->host_ref(), MaterialId{0});
    MevEnergy mean_loss{0.1};

    int num_samples = 5000;
    std::vector<real_type> mean;
    std::vector<real_type> counts(20);
    real_type upper = 7.0;
    real_type lower = 0.0;
    real_type width = (upper - lower) / counts.size();

    // Larger step samples from gamma distribution, smaller step from Gaussian
    {
        real_type sum = 0;
        real_type step = 5e-2 * units::centimeter;
        EnergyLossHelper helper(
            fluct->host_ref(), cutoff, material, particle, mean_loss, step);
        EXPECT_EQ(EnergyLossFluctuationModel::gamma, helper.model());
        EXPECT_SOFT_EQ(0.00019160444039613,
                       value_as<MevEnergy>(helper.max_energy()));
        EXPECT_SOFT_EQ(0.00018926243294348, helper.beta_sq());
        EXPECT_SOFT_EQ(0.13988041753438,
                       value_as<EnergySq>(helper.bohr_variance()));

        EnergyLossGammaDistribution sample_loss(helper);
        for ([[maybe_unused]] int i : range(num_samples))
        {
            auto loss = sample_loss(rng).value();
            EXPECT_GE(loss, 0);
            EXPECT_LT(loss, upper);
            auto bin = static_cast<size_type>((loss - lower) / width);
            CELER_ASSERT(bin < counts.size());
            counts[bin]++;
            sum += loss;
        }
        EXPECT_SOFT_EQ(0.096429312200727382, sum / num_samples);
    }
    {
        real_type sum = 0;
        real_type step = 5e-4 * units::centimeter;
        EnergyLossHelper helper(
            fluct->host_ref(), cutoff, material, particle, mean_loss, step);
        EXPECT_SOFT_EQ(0.00019160444039613,
                       value_as<MevEnergy>(helper.max_energy()));
        EXPECT_SOFT_EQ(0.00018926243294348, helper.beta_sq());
        EXPECT_SOFT_EQ(0.0013988041753438,
                       value_as<EnergySq>(helper.bohr_variance()));
        EXPECT_EQ(EnergyLossFluctuationModel::gaussian, helper.model());

        EnergyLossGaussianDistribution sample_loss(helper);
        for ([[maybe_unused]] int i : range(num_samples))
        {
            auto loss = sample_loss(rng).value();
            auto bin = size_type((loss - lower) / width);
            CELER_ASSERT(bin < counts.size());
            counts[bin]++;
            sum += loss;
        }
        EXPECT_SOFT_EQ(0.10031120242856037, sum / num_samples);
    }

    static real_type const expected_counts[] = {
        9636, 166, 85, 31, 27, 18, 13, 6, 6, 4, 2, 2, 0, 3, 0, 0, 1, 0, 0, 0};
    EXPECT_VEC_SOFT_EQ(expected_counts, counts);
    EXPECT_EQ(41006, rng.count());
}

TEST_F(EnergyLossDistributionTest, urban)
{
    ParticleTrackView particle(
        particles->host_ref(), particle_state.ref(), TrackSlotId{0});
    particle = {ParticleId{0}, MevEnergy{100}};
    MaterialTrackView material(
        materials->host_ref(), material_state.ref(), TrackSlotId{0});
    material = {MaterialId{0}};
    CutoffView cutoff(cutoffs->host_ref(), MaterialId{0});
    MevEnergy mean_loss{0.01};
    real_type step = 0.01 * units::centimeter;

    int num_samples = 10000;
    std::vector<real_type> counts(20);
    real_type upper = 0.03;
    real_type lower = 0.0;
    real_type width = (upper - lower) / counts.size();
    real_type sum = 0;

    EnergyLossHelper helper(
        fluct->host_ref(), cutoff, material, particle, mean_loss, step);
    EXPECT_SOFT_EQ(0.001, value_as<MevEnergy>(helper.max_energy()));
    EXPECT_SOFT_EQ(0.99997415284006, helper.beta_sq());
    EXPECT_SOFT_EQ(1.3819085992495e-05,
                   value_as<EnergySq>(helper.bohr_variance()));
    EXPECT_EQ(EnergyLossFluctuationModel::urban, helper.model());
    EnergyLossUrbanDistribution sample_loss(helper);

    for ([[maybe_unused]] int i : range(num_samples))
    {
        auto loss = sample_loss(rng).value();
        auto bin = size_type((loss - lower) / width);
        CELER_ASSERT(bin < counts.size());
        counts[bin]++;
        sum += loss;
    }
#ifdef _MSC_VER
    // TODO: determine why the sampled sequence is different
    GTEST_SKIP() << "Results differ statistically when built with MSVC...";
#endif

    static real_type const expected_counts[]
        = {0,   0,   12, 223, 1174, 2359, 2661, 1867, 898, 369,
           189, 107, 60, 48,  20,   9,    2,    2,    0,   0};
    EXPECT_VEC_SOFT_EQ(expected_counts, counts);
    EXPECT_SOFT_EQ(0.0099918954960280353, sum / num_samples);
    EXPECT_EQ(551188, rng.count());
}
//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
