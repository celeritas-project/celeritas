//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/distribution/EnergyLossHelper.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/em/distribution/EnergyLossHelper.hh"

#include "corecel/data/CollectionStateStore.hh"
#include "corecel/random/HistogramSampler.hh"
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

#include "TestMacros.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
using units::MevEnergy;
using EnergySq = RealQuantity<UnitProduct<units::Mev, units::Mev>>;

real_type to_mev(MevEnergy e)
{
    return e.value();
}

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
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(MockFluctuationTest, data)
{
    auto const& urban = fluct->host_ref().urban;

    {
        // Celerogen: Z=1, I=19.2 eV
        auto const& params = urban[PhysMatId{0}];
        EXPECT_SOFT_EQ(1, params.oscillator_strength[0]);
        EXPECT_SOFT_EQ(0, params.oscillator_strength[1]);
        EXPECT_SOFT_EQ(19.2e-6, params.binding_energy[0]);
        EXPECT_SOFT_EQ(1e-5, params.binding_energy[1]);
    }
    {
        // Celer composite: Z_eff = 10.3, I=150.7 eV
        auto const& params = urban[PhysMatId{2}];
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
    material = {PhysMatId{0}};
    CutoffView cutoff(cutoffs->host_ref(), PhysMatId{0});
    MevEnergy mean_loss{2e-6};

    // Tiny step, little energy loss
    real_type step = 1e-6 * units::centimeter;
    EnergyLossHelper helper(
        fluct->host_ref(), cutoff, material, particle, mean_loss, step);
    EXPECT_EQ(EnergyLossFluctuationModel::none, helper.model());

    DiagnosticRngEngine<std::mt19937> rng;
    EnergyLossDeltaDistribution sample_loss(helper);
    EXPECT_EQ(mean_loss, sample_loss(rng));
    EXPECT_EQ(0, rng.exchange_count());
}

TEST_F(EnergyLossDistributionTest, gaussian)
{
    ParticleTrackView particle(
        particles->host_ref(), particle_state.ref(), TrackSlotId{0});
    particle = {ParticleId{1}, MevEnergy{1e-2}};
    MaterialTrackView material(
        materials->host_ref(), material_state.ref(), TrackSlotId{0});
    material = {PhysMatId{0}};
    CutoffView cutoff(cutoffs->host_ref(), PhysMatId{0});
    MevEnergy mean_loss{0.1};

    // Larger step samples from gamma distribution, smaller step from Gaussian
    {
        real_type step = 5e-2 * units::centimeter;
        EnergyLossHelper helper(
            fluct->host_ref(), cutoff, material, particle, mean_loss, step);
        EXPECT_EQ(EnergyLossFluctuationModel::gamma, helper.model());
        EXPECT_SOFT_EQ(0.00019160444039613,
                       value_as<MevEnergy>(helper.max_energy()));
        EXPECT_SOFT_EQ(0.00018926243294348, helper.beta_sq());
        EXPECT_SOFT_EQ(0.13988041753438,
                       value_as<EnergySq>(helper.bohr_variance()));

        HistogramSampler calc_histogram(21, {0, 7}, 10000);
        auto sampled
            = calc_histogram(to_mev, EnergyLossGammaDistribution{helper});
        SampledHistogram ref;
        ref.distribution = {
            2.7684, 0.105,  0.0507, 0.0225, 0.0168, 0.0108, 0.0078,
            0.006,  0.0042, 0.0033, 0.0012, 0.0006, 0,      0.0003,
            0.0009, 0.0003, 0.0006, 0.0006, 0,      0,      0,
        };
        ref.rng_count = 6.1764;
        EXPECT_REF_EQ(ref, sampled);
    }
    {
        real_type step = 5e-4 * units::centimeter;
        EnergyLossHelper helper(
            fluct->host_ref(), cutoff, material, particle, mean_loss, step);
        EXPECT_SOFT_EQ(0.00019160444039613,
                       value_as<MevEnergy>(helper.max_energy()));
        EXPECT_SOFT_EQ(0.00018926243294348, helper.beta_sq());
        EXPECT_SOFT_EQ(0.0013988041753438,
                       value_as<EnergySq>(helper.bohr_variance()));
        EXPECT_EQ(EnergyLossFluctuationModel::gaussian, helper.model());

        HistogramSampler calc_histogram(16, {0, 0.2}, 10000);
        auto sampled
            = calc_histogram(to_mev, EnergyLossGaussianDistribution{helper});
        // sampled.print_expected();
        SampledHistogram ref;
        ref.distribution = {
            0.32,
            1.256,
            2.008,
            3.512,
            5.568,
            7.64,
            9.256,
            10.504,
            10.68,
            9.44,
            7.456,
            5.12,
            3.576,
            2.136,
            1.064,
            0.464,
        };
        ref.rng_count = 2.0148;
        EXPECT_REF_EQ(ref, sampled) << sampled;
    }
}

TEST_F(EnergyLossDistributionTest, urban)
{
    ParticleTrackView particle(
        particles->host_ref(), particle_state.ref(), TrackSlotId{0});
    particle = {ParticleId{0}, MevEnergy{100}};
    MaterialTrackView material(
        materials->host_ref(), material_state.ref(), TrackSlotId{0});
    material = {PhysMatId{0}};
    CutoffView cutoff(cutoffs->host_ref(), PhysMatId{0});
    MevEnergy mean_loss{0.01};
    real_type step = 0.01 * units::centimeter;

    EnergyLossHelper helper(
        fluct->host_ref(), cutoff, material, particle, mean_loss, step);
    EXPECT_SOFT_EQ(0.001, value_as<MevEnergy>(helper.max_energy()));
    EXPECT_SOFT_EQ(0.99997415284006, helper.beta_sq());
    EXPECT_SOFT_EQ(1.3819085992495e-05,
                   value_as<EnergySq>(helper.bohr_variance()));
    EXPECT_EQ(EnergyLossFluctuationModel::urban, helper.model());

    HistogramSampler calc_histogram(15, {0, 0.03}, 10000);
    auto sampled = calc_histogram(to_mev, EnergyLossUrbanDistribution{helper});
#ifdef _MSC_VER
    // TODO: determine why the sampled sequence is different
    GTEST_SKIP() << "Results differ statistically when built with MSVC...";
#endif
    SampledHistogram ref;
    ref.distribution = {
        0,
        0.2,
        11.55,
        95.35,
        173,
        134.7,
        52.65,
        17.55,
        7.95,
        4,
        2.25,
        0.6,
        0.2,
        0,
        0,
    };
    ref.rng_count = 55.1188;
    EXPECT_REF_EQ(ref, sampled);
}
//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
