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
    // Delta rays can only be produced above the minimum energy
    // (5 keV scaled by the muon/proton mass ratio)
    for (real_type energy : {0.05, 0.1, 0.2, 0.5, 1.0})
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
        {2.35609926262,
         2.1773492834392,
         2.0530668905071,
         1.9260761645571,
         1.8106710854058,
         1.6903007290551,
         1.5566896335058,
         1.4760414947508},
        {1.3244614220788,
         1.1151022208568,
         0.97078411895389,
         0.83136369019252,
         0.71414604633928,
         0.60176077331899,
         0.52424693306618,
         0.4483656505272},
        {1.0480941749032,
         0.81297450858349,
         0.64515101914619,
         0.51268512450214,
         0.39706412171713,
         0.31071320330107,
         0.25012996599612,
         0.19271556973397},
        {0.89305904844558,
         0.62959492402496,
         0.4421154212925,
         0.30921284616457,
         0.21134104317351,
         0.15484616669192,
         0.10455641893023,
         0.075787185774351},
        {0.83290697177157,
         0.53617635987068,
         0.34785435167649,
         0.21849339560275,
         0.14311022946487,
         0.091080442019349,
         0.056877142479628,
         0.038620281930612},
    };
    static double const expected_min_energy[] = {
        0.00056304762676649,
        0.00056304762676649,
        0.00056304762676649,
        0.00056304762676649,
        0.00056304762676649,
    };
    static double const expected_max_energy[] = {
        0.00095820006165405,
        0.001916844768863,
        0.0038354680957569,
        0.0096020089408745,
        0.019248476995285,
    };
    EXPECT_VEC_SOFT_EQ(expected_loge_pdf, loge_pdf);
    EXPECT_VEC_SOFT_EQ(expected_min_energy, min_energy);
    EXPECT_VEC_SOFT_EQ(expected_max_energy, max_energy);
}

TEST_F(BraggICRU73QOTest, min_energy)
{
    int const num_samples = 10000;
    real_type const mass_ratio = 105.6583755 / 938.27208816;

    for (auto pdg : {pdg::mu_minus(), pdg::mu_plus()})
    {
        SCOPED_TRACE(pdg == pdg::mu_minus() ? "mu-" : "mu+");
        this->set_inc_particle(pdg, MevEnergy{1});
        std::mt19937 rng;

        // Delta rays are not produced below the production cut
        MevEnergy cutoff{1e-3};
        BraggICRU73QOEnergyDistribution sample(
            this->particle_track(),
            cutoff,
            bragg_model_->host_ref().electron_mass);
        EXPECT_SOFT_EQ(cutoff.value(), sample.min_secondary_energy().value());
        real_type min_sampled = sample.max_secondary_energy().value();
        for ([[maybe_unused]] int i : range(num_samples))
        {
            min_sampled = std::fmin(min_sampled, sample(rng).value());
        }
        EXPECT_LE(cutoff.value(), min_sampled);

        // A small cut is limited by the scaled lowest model energy
        BraggICRU73QOEnergyDistribution sample_low(
            this->particle_track(),
            MevEnergy{1e-6},
            bragg_model_->host_ref().electron_mass);
        EXPECT_SOFT_NEAR((pdg == pdg::mu_minus() ? 5e-3 : 2.5e-4) * mass_ratio,
                         sample_low.min_secondary_energy().value(),
                         1e-6);
    }
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

        static double const expected_energy[] = {
            0.0017050572502434,
            0.0015938457100761,
            0.0016345330723376,
            0.0010156943192646,
        };
        static double const expected_costheta[] = {
            0.94222194263696,
            0.91051113884193,
            0.92223165280284,
            0.7249299995025,
        };

        EXPECT_VEC_SOFT_EQ(expected_energy, energy);
        EXPECT_VEC_SOFT_EQ(expected_costheta, costheta);
    }
    // Sample Bragg model with incident mu+
    {
        inc_particle_ = pdg::mu_plus();
        sample(bragg_model_->host_ref());

        static double const expected_energy[] = {
            0.0017414365337993,
            0.0019084747469908,
            0.001061737665425,
            0.0014398828372545,
        };
        static double const expected_costheta[] = {
            0.95237965465919,
            0.99777589611165,
            0.74133463089421,
            0.86480734182311,
        };
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

        // Max delta ray energy must be above the 1 keV production cut
        for (real_type inc_e : {0.06, 0.08, 0.1, 0.15, 0.1999})
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
            avg_engine_samples.push_back(
                real_type(rng.count()) / num_particles_sampled);
            avg_energy.push_back(energy / num_particles_sampled);
            avg_costheta.push_back(costheta / num_particles_sampled);
        }
    };

    // Sample ICRU73QO model with incident mu-
    {
        inc_particle_ = pdg::mu_minus();
        sample(icru73qo_model_->host_ref());

        static double const expected_avg_engine_samples[] = {
            6.0048,
            6.0031,
            6.005,
            6.0053,
            6.0088,
        };
        static double const expected_avg_energy[] = {
            0.0010711185435677,
            0.0012280881510912,
            0.0013583013424647,
            0.0016199441868758,
            0.0018147430687587,
        };
        static double const expected_avg_costheta[] = {
            0.96434396889379,
            0.89171055862578,
            0.83604230697034,
            0.73947035170344,
            0.67297240869032,
        };
        EXPECT_VEC_SOFT_EQ(expected_avg_engine_samples, avg_engine_samples);
        EXPECT_VEC_SOFT_EQ(expected_avg_energy, avg_energy);
        EXPECT_VEC_SOFT_EQ(expected_avg_costheta, avg_costheta);
    }
    // Sample Bragg model with incident mu+
    {
        inc_particle_ = pdg::mu_plus();
        sample(bragg_model_->host_ref());

        static double const expected_avg_engine_samples[] = {
            6.0031,
            6.0059,
            6.0059,
            6.0065,
            6.0068,
        };
        static double const expected_avg_energy[] = {
            0.0010715288007516,
            0.0012295968310086,
            0.0013584646761927,
            0.0016196110791739,
            0.0018077256180413,
        };
        static double const expected_avg_costheta[] = {
            0.9645299202306,
            0.89225051248253,
            0.83612344972909,
            0.73945524322875,
            0.67170643765452,
        };

        EXPECT_VEC_SOFT_EQ(expected_avg_engine_samples, avg_engine_samples);
        EXPECT_VEC_SOFT_EQ(expected_avg_energy, avg_energy);
        EXPECT_VEC_SOFT_EQ(expected_avg_costheta, avg_costheta);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
