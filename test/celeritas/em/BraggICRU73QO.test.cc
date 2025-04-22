//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/BraggICRU73QO.test.cc
//---------------------------------------------------------------------------//
#include "corecel/cont/Range.hh"
#include "corecel/math/ArrayUtils.hh"
#include "corecel/random/Histogram.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/em/distribution/BraggICRU73QOEnergyDistribution.hh"
#include "celeritas/em/interactor/MuHadIonizationInteractor.hh"
#include "celeritas/em/interactor/detail/PhysicsConstants.hh"
#include "celeritas/em/model/BraggModel.hh"
#include "celeritas/em/model/ICRU73QOModel.hh"
#include "celeritas/em/process/MuIonizationProcess.hh"
#include "celeritas/phys/CutoffView.hh"
#include "celeritas/phys/InteractionIO.hh"
#include "celeritas/phys/InteractorHostTestBase.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class BraggICRU73QOTest : public InteractorHostTestBase
{
    using Base = InteractorHostTestBase;

  protected:
    void SetUp() override
    {
        using namespace units;

        // Set up shared material data
        MaterialParams::Input mat_inp;
        mat_inp.elements = {{AtomicNumber{29}, AmuMass{63.546}, {}, "Cu"}};
        mat_inp.materials = {
            {native_value_from(MolCcDensity{0.141}),
             293.0,
             MatterState::solid,
             {{ElementId{0}, 1.0}},
             "Cu"},
        };
        this->set_material_params(mat_inp);

        // Set 1 keV electron cutoff
        CutoffParams::Input cut_inp;
        cut_inp.materials = this->material_params();
        cut_inp.particles = this->particle_params();
        cut_inp.cutoffs = {{pdg::electron(), {{MevEnergy{0.001}, 0.1234}}}};
        this->set_cutoff_params(cut_inp);

        auto const& particles = *this->particle_params();

        // Set ICRU73QO model data
        Applicability mu_minus;
        mu_minus.particle = particles.find(pdg::mu_minus());
        mu_minus.lower = zero_quantity();
        mu_minus.upper
            = MuIonizationProcess::Options{}.bragg_icru73qo_upper_limit;
        icru73qo_model_ = std::make_shared<ICRU73QOModel>(
            ActionId{0}, particles, Model::SetApplicability{mu_minus});

        // Set Bragg model data
        Applicability mu_plus = mu_minus;
        mu_plus.particle = particles.find(pdg::mu_plus());
        bragg_model_ = std::make_shared<BraggModel>(
            ActionId{0}, particles, Model::SetApplicability{mu_plus});

        // Set default particle to muon with energy of 100 keV
        inc_particle_ = pdg::mu_minus();
        this->set_inc_particle(inc_particle_, MevEnergy{0.1});
        this->set_inc_direction({0, 0, 1});
        this->set_material("Cu");
    }

    void sanity_check(Interaction const& interaction) const
    {
        // Check change to parent track
        EXPECT_GT(this->particle_track().energy().value(),
                  interaction.energy.value());
        EXPECT_LT(0, interaction.energy.value());
        EXPECT_SOFT_EQ(1.0, norm(interaction.direction));
        EXPECT_EQ(Action::scattered, interaction.action);

        // Check secondaries
        ASSERT_EQ(1, interaction.secondaries.size());

        auto const& electron = interaction.secondaries.front();
        EXPECT_TRUE(electron);
        EXPECT_EQ(bragg_model_->host_ref().electron, electron.particle_id);
        EXPECT_GT(this->particle_track().energy().value(),
                  electron.energy.value());
        EXPECT_LT(0, electron.energy.value());
        EXPECT_SOFT_EQ(1.0, norm(electron.direction));

        // Check conservation between primary and secondaries
        this->check_conservation(interaction);
        this->check_energy_conservation(interaction);
    }

  protected:
    std::shared_ptr<BraggModel> bragg_model_;
    std::shared_ptr<ICRU73QOModel> icru73qo_model_;
    PDGNumber inc_particle_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(BraggICRU73QOTest, distribution)
{
    int num_samples = 100000;
    int num_bins = 8;

    MevEnergy cutoff{1e-6};

    std::vector<std::vector<double>> loge_pdf;
    std::vector<real_type> min_energy;
    std::vector<real_type> max_energy;
    for (real_type energy : {1e-4, 1e-3, 1e-2, 0.1, 0.2, 0.5, 1.0})
    {
        this->set_inc_particle(pdg::mu_minus(), MevEnergy(energy));
        std::mt19937 rng;

        BraggICRU73QOEnergyDistribution sample(
            this->particle_track(),
            cutoff,
            bragg_model_->host_ref().electron_mass);
        real_type min = value_as<MevEnergy>(sample.min_secondary_energy());
        real_type max = value_as<MevEnergy>(sample.max_secondary_energy());

        Histogram histogram(num_bins, {std::log(min), std::log(max)});
        for ([[maybe_unused]] int i : range(num_samples))
        {
            histogram(std::log(value_as<MevEnergy>(sample(rng))));
        }
        EXPECT_FALSE(histogram.underflow() || histogram.overflow());
        loge_pdf.push_back(histogram.calc_density());
        min_energy.push_back(min);
        max_energy.push_back(max);
    }

    static std::vector<double> const expected_loge_pdf[] = {
        {2.0156965951408,
         1.8360642305613,
         1.711428898151,
         1.5743669432596,
         1.4654802017165,
         1.3377689726525,
         1.2237147247312,
         1.1390660488197},
        {0.88133112913034,
         0.61034822247305,
         0.42356996227555,
         0.29097744626068,
         0.19853656668513,
         0.14145188518873,
         0.095258538273049,
         0.067813458844551},
        {0.73703536322629,
         0.38285712793805,
         0.19676399489989,
         0.10316181289158,
         0.052517080489295,
         0.027339326538775,
         0.014887450643052,
         0.0076720604540883},
        {0.64723444809841,
         0.25145943677,
         0.098030865534292,
         0.037531575165688,
         0.014733207171641,
         0.0063187677309406,
         0.0022967715202917,
         0.00081498344268414},
        {0.62491163061924,
         0.22117179554681,
         0.079921990115187,
         0.027493785051755,
         0.010421656886684,
         0.0038002693019352,
         0.0012796825200394,
         0.00045564453365039},
        {0.59540263638262,
         0.18871661969335,
         0.060014868793425,
         0.019149969036425,
         0.0064124042103746,
         0.0019978783186065,
         0.00054963464660354,
         0.00019193590833774},
        {0.57500784058224,
         0.16731968317279,
         0.048193712843305,
         0.014158879795459,
         0.0045817680895957,
         0.001232617255962,
         0.00031626363804289,
         0.00012163986078573},
    };
    static double const expected_min_energy[]
        = {1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06};
    static double const expected_max_energy[] = {
        1.9159563630249e-06,
        1.915964366753e-05,
        0.00019160444039615,
        0.001916844768863,
        0.0038354680957569,
        0.0096020089408745,
        0.019248476995285,
    };
    EXPECT_VEC_SOFT_EQ(expected_loge_pdf, loge_pdf);
    EXPECT_VEC_SOFT_EQ(expected_min_energy, min_energy);
    EXPECT_VEC_SOFT_EQ(expected_max_energy, max_energy);
}

TEST_F(BraggICRU73QOTest, basic)
{
    std::vector<real_type> energy;
    std::vector<real_type> costheta;

    auto sample = [&](MuHadIonizationData const& data) {
        // Reserve 4 secondaries, one for each sample
        int const num_samples = 4;
        this->resize_secondaries(num_samples);

        energy.clear();
        costheta.clear();

        this->set_inc_particle(inc_particle_, MevEnergy{0.1});

        // Create the interactor
        MuHadIonizationInteractor<BraggICRU73QOEnergyDistribution> interact(
            data,
            this->particle_track(),
            this->cutoff_params()->get(PhysMatId{0}),
            this->direction(),
            this->secondary_allocator());
        RandomEngine& rng = this->rng();

        // Produce four samples from the original incident energy
        for (int i : range(num_samples))
        {
            Interaction result = interact(rng);
            SCOPED_TRACE(result);
            this->sanity_check(result);

            EXPECT_EQ(result.secondaries.data(),
                      this->secondary_allocator().get().data() + i);

            energy.push_back(result.secondaries.front().energy.value());
            costheta.push_back(dot_product(
                result.direction, result.secondaries.front().direction));
        }

        EXPECT_EQ(num_samples, this->secondary_allocator().get().size());

        // Next sample should fail because we're out of secondary buffer space
        {
            Interaction result = interact(rng);
            EXPECT_EQ(0, result.secondaries.size());
            EXPECT_EQ(Action::failed, result.action);
        }

        // No interaction when max secondary energy is below production cut
        {
            this->set_inc_particle(inc_particle_, MevEnergy{0.0011});
            MuHadIonizationInteractor<BraggICRU73QOEnergyDistribution> interact(
                data,
                this->particle_track(),
                this->cutoff_params()->get(PhysMatId{0}),
                this->direction(),
                this->secondary_allocator());

            Interaction result = interact(rng);
            EXPECT_EQ(0, result.secondaries.size());
            EXPECT_EQ(Action::unchanged, result.action);
        }
    };

    // Sample ICRU73QO model with incident mu-
    {
        inc_particle_ = pdg::mu_minus();
        sample(icru73qo_model_->host_ref());

        static double const expected_energy[] = {0.0014458653777536,
                                                 0.001251648293082,
                                                 0.0013192801865397,
                                                 0.00057619400045627};
        static double const expected_costheta[] = {0.86662579730412,
                                                   0.80560684873176,
                                                   0.82734134051617,
                                                   0.54491853032358};

        EXPECT_VEC_SOFT_EQ(expected_energy, energy);
        EXPECT_VEC_SOFT_EQ(expected_costheta, costheta);
    }
    // Sample Bragg model with incident mu+
    {
        inc_particle_ = pdg::mu_plus();
        sample(bragg_model_->host_ref());

        static double const expected_energy[] = {0.00022900204776481,
                                                 0.0014511488605566,
                                                 3.1983487781218e-05,
                                                 7.5949049601834e-05};
        static double const expected_costheta[] = {0.3429925946801,
                                                   0.86822876526498,
                                                   0.12806842199932,
                                                   0.19739055297736};
        EXPECT_VEC_SOFT_EQ(expected_energy, energy);
        EXPECT_VEC_SOFT_EQ(expected_costheta, costheta);
    }
}

TEST_F(BraggICRU73QOTest, stress_test)
{
    std::vector<real_type> avg_engine_samples;
    std::vector<real_type> avg_energy;
    std::vector<real_type> avg_costheta;

    auto sample = [&](MuHadIonizationData const& data) {
        unsigned int const num_samples = 10000;

        avg_engine_samples.clear();
        avg_energy.clear();
        avg_costheta.clear();

        for (real_type inc_e : {0.03, 0.05, 0.1, 0.15, 0.1999})
        {
            SCOPED_TRACE("Incident energy: " + std::to_string(inc_e));
            this->set_inc_particle(inc_particle_, MevEnergy{inc_e});

            RandomEngine& rng = this->rng();
            RandomEngine::size_type num_particles_sampled = 0;
            real_type energy = 0;
            real_type costheta = 0;

            // Loop over several incident directions
            for (Real3 const& inc_dir : {Real3{0, 0, 1},
                                         Real3{1, 0, 0},
                                         Real3{1e-9, 0, 1},
                                         Real3{1, 1, 1}})
            {
                SCOPED_TRACE("Incident direction: " + to_string(inc_dir));
                this->set_inc_direction(inc_dir);
                this->resize_secondaries(num_samples);

                // Create interactor
                MuHadIonizationInteractor<BraggICRU73QOEnergyDistribution>
                    interact(data,
                             this->particle_track(),
                             this->cutoff_params()->get(PhysMatId{0}),
                             this->direction(),
                             this->secondary_allocator());

                // Loop over many particles
                for (unsigned int i = 0; i < num_samples; ++i)
                {
                    Interaction result = interact(rng);
                    this->sanity_check(result);

                    energy += result.secondaries.front().energy.value();
                    costheta += dot_product(
                        result.direction, result.secondaries.front().direction);
                }
                EXPECT_EQ(num_samples,
                          this->secondary_allocator().get().size());
                num_particles_sampled += num_samples;
            }
            avg_engine_samples.push_back(real_type(rng.count())
                                         / num_particles_sampled);
            avg_energy.push_back(energy / num_particles_sampled);
            avg_costheta.push_back(costheta / num_particles_sampled);
        }
    };

    // Sample ICRU73QO model with incident mu-
    {
        inc_particle_ = pdg::mu_minus();
        sample(icru73qo_model_->host_ref());

        static double const expected_avg_engine_samples[]
            = {6.0027, 6.0021, 6.003, 6.0034, 6.0047};
        static double const expected_avg_energy[] = {0.00056893310178363,
                                                     0.00072492000606412,
                                                     0.00097487866369081,
                                                     0.0011443155397256,
                                                     0.0012633301124393};
        static double const expected_avg_costheta[] = {0.99472414540371,
                                                       0.86537547828047,
                                                       0.69947394093299,
                                                       0.6113577317816,
                                                       0.55099275303428};
        EXPECT_VEC_SOFT_EQ(expected_avg_engine_samples, avg_engine_samples);
        EXPECT_VEC_SOFT_EQ(expected_avg_energy, avg_energy);
        EXPECT_VEC_SOFT_EQ(expected_avg_costheta, avg_costheta);
    }
    // Sample Bragg model with incident mu+
    {
        inc_particle_ = pdg::mu_plus();
        sample(bragg_model_->host_ref());

        static double const expected_avg_engine_samples[]
            = {6.0004, 6.0004, 6.0006, 6.0003, 6.0005};
        static double const expected_avg_energy[] = {8.8601911130921e-05,
                                                     0.00010210349779604,
                                                     0.00012023485252326,
                                                     0.00013103967324893,
                                                     0.00013806656798748};
        static double const expected_avg_costheta[] = {0.35858206068691,
                                                       0.29003901277676,
                                                       0.21408617527108,
                                                       0.17851819065736,
                                                       0.15626495143414};

        EXPECT_VEC_SOFT_EQ(expected_avg_engine_samples, avg_engine_samples);
        EXPECT_VEC_SOFT_EQ(expected_avg_energy, avg_energy);
        EXPECT_VEC_SOFT_EQ(expected_avg_costheta, avg_costheta);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
