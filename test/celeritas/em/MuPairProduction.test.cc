//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/MuPairProduction.test.cc
//---------------------------------------------------------------------------//
#include "corecel/cont/Range.hh"
#include "corecel/math/ArrayUtils.hh"
#include "corecel/random/Histogram.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/RootTestBase.hh"
#include "celeritas/em/distribution/MuPPEnergyDistribution.hh"
#include "celeritas/em/interactor/MuPairProductionInteractor.hh"
#include "celeritas/em/model/MuPairProductionModel.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/mat/MaterialTrackView.hh"
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

class MuPairProductionTest : public InteractorHostBase,  public RootTestBase
{
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

        // Set 1 keV cutoff
        CutoffParams::Input cut_inp;
        cut_inp.materials = this->material_params();
        cut_inp.particles = this->particle_params();
        cut_inp.cutoffs = {{pdg::positron(), {{MevEnergy{0.001}, 0.1234}}}};
        this->set_cutoff_params(cut_inp);

        // Construct model
        auto imported = std::make_shared<ImportedProcesses>(
            this->imported_data().processes);
        model_ = std::make_shared<MuPairProductionModel>(
            ActionId{0},
            *this->particle_params(),
            imported,
            this->imported_data().mu_pair_production_data);

        // Set default particle to 10 GeV muon
        this->set_inc_particle(pdg::mu_minus(), MevEnergy{1e4});
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
        ASSERT_EQ(2, interaction.secondaries.size());
        auto const& electron = interaction.secondaries[0];
        EXPECT_TRUE(electron);
        EXPECT_GT(this->particle_track().energy(), electron.energy);
        EXPECT_LT(zero_quantity(), electron.energy);
        EXPECT_SOFT_EQ(1.0, norm(electron.direction));
        EXPECT_EQ(model_->host_ref().ids.electron, electron.particle_id);

        auto const& positron = interaction.secondaries[1];
        EXPECT_TRUE(positron);
        EXPECT_GT(this->particle_track().energy(), positron.energy);
        EXPECT_LT(zero_quantity(), positron.energy);
        EXPECT_SOFT_EQ(1.0, norm(positron.direction));
        EXPECT_EQ(model_->host_ref().ids.positron, positron.particle_id);

        // Check conservation between primary and secondaries
        // this->check_conservation(interaction);
        this->check_energy_conservation(interaction);
    }

    //! \note These tests use a trimmed element table
    std::string_view geometry_basename() const final
    {
        return "four-steel-slabs";
    }

    SPConstTrackInit build_init() override { CELER_ASSERT_UNREACHABLE(); }
    SPConstAction build_along_step() override { CELER_ASSERT_UNREACHABLE(); }

  protected:
    std::shared_ptr<MuPairProductionModel> model_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(MuPairProductionTest, distribution)
{
    int num_samples = 10000;
    int num_bins = 8;

    real_type two_me
        = 2 * value_as<units::MevMass>(model_->host_ref().electron_mass);

    // Get view to the current element
    auto element = this->material_track().material_record().element_record(
        ElementComponentId{0});

    // Get the production cuts
    auto cutoff = this->cutoff_params()->get(MaterialId{0});

    RandomEngine& rng = InteractorHostBase::rng();

    std::vector<real_type> loge_pdf;
    std::vector<real_type> min_energy;
    std::vector<real_type> max_energy;
    std::vector<real_type> avg_energy;
    std::vector<real_type> avg_energy_fraction;
    for (real_type energy : {1e3, 1e4, 1e5, 1e6, 1e7})
    {
        this->set_inc_particle(pdg::mu_minus(), MevEnergy(energy));

        MuPPEnergyDistribution sample(
            model_->host_ref(), this->particle_track(), cutoff, element);
        real_type min = value_as<MevEnergy>(sample.min_pair_energy()) - two_me;
        real_type max = value_as<MevEnergy>(sample.max_pair_energy()) - two_me;

        real_type sum_energy = 0;
        real_type energy_fraction = 0;
        Histogram histogram(num_bins, {std::log(min), std::log(max)});
        for ([[maybe_unused]] int i : range(num_samples))
        {
            // TODO: test energy partition
            auto e = sample(rng);
            auto e_pair = value_as<MevEnergy>(e.electron + e.positron);
            ASSERT_GE(e_pair, min);
            ASSERT_LE(e_pair, max);
            histogram(std::log(e_pair));
            sum_energy += e_pair;
            energy_fraction += value_as<MevEnergy>(e.electron) / e_pair;
        }
        auto density = histogram.density();
        loge_pdf.insert(loge_pdf.end(), density.begin(), density.end());
        min_energy.push_back(min);
        max_energy.push_back(max);
        avg_energy.push_back(sum_energy / num_samples);
        avg_energy_fraction.push_back(energy_fraction / num_samples);
    }

    static double const expected_loge_pdf[] = {
        0.0486, 0.2855, 0.3831, 0.2029, 0.0631, 0.015,  0.0016, 0.0002,
        0.0639, 0.2435, 0.3676, 0.2433, 0.0685, 0.0112, 0.002,  0,
        0.053,  0.2099, 0.3242, 0.267,  0.1219, 0.0215, 0.0023, 0.0002,
        0.0522, 0.2027, 0.3008, 0.2712, 0.1369, 0.0338, 0.0022, 0.0002,
        0.0533, 0.1979, 0.2939, 0.2582, 0.1485, 0.0435, 0.0046, 0.0001,
    };
    static double const expected_min_energy[] = {
        1.0219978922,
        1.0219978922,
        1.0219978922,
        1.0219978922,
        1.0219978922,
    };
    static double const expected_max_energy[] = {
        703.23539643546,
        9703.2353964355,
        99703.235396435,
        999703.23539644,
        9999703.2353964,
    };
    static double const expected_avg_energy[] = {
        11.634922704826,
        42.584416898446,
        216.27235630244,
        1093.1529390214,
        6041.4317155177,
    };
    static double const expected_avg_energy_fraction[] = {
        0.50427657004076,
        0.5011248037151,
        0.49759910105122,
        0.50543111979394,
        0.50102592402615,
    };
    EXPECT_VEC_SOFT_EQ(expected_loge_pdf, loge_pdf);
    EXPECT_VEC_SOFT_EQ(expected_min_energy, min_energy);
    EXPECT_VEC_SOFT_EQ(expected_max_energy, max_energy);
    EXPECT_VEC_SOFT_EQ(expected_avg_energy, avg_energy);
    EXPECT_VEC_SOFT_EQ(expected_avg_energy_fraction, avg_energy_fraction);
}

TEST_F(MuPairProductionTest, basic)
{
    // Reserve 8 secondaries, two for each sample
    int const num_samples = 4;
    this->resize_secondaries(2 * num_samples);

    // Get view to the current element
    auto element = this->material_track().material_record().element_record(
        ElementComponentId{0});

    // Get the production cuts
    auto cutoff = this->cutoff_params()->get(MaterialId{0});

    // Create the interactor
    MuPairProductionInteractor interact(model_->host_ref(),
                                        this->particle_track(),
                                        cutoff,
                                        element,
                                        this->direction(),
                                        this->secondary_allocator());
    RandomEngine& rng = InteractorHostBase::rng();

    std::vector<real_type> pair_energy;
    std::vector<real_type> costheta;

    // Produce four samples from the original incident energy
    for (int i : range(num_samples))
    {
        Interaction result = interact(rng);
        SCOPED_TRACE(result);
        this->sanity_check(result);

        EXPECT_EQ(result.secondaries.data(),
                  this->secondary_allocator().get().data()
                      + result.secondaries.size() * i);

        pair_energy.push_back(value_as<MevEnergy>(
            result.secondaries[0].energy + result.secondaries[1].energy));
        costheta.push_back(dot_product(result.secondaries[0].direction,
                                       result.secondaries[1].direction));
    }

    EXPECT_EQ(2 * num_samples, this->secondary_allocator().get().size());

    // Note: these are "gold" values based on the host RNG.
    static double const expected_pair_energy[] = {
        5.1919218572645,
        21.387748984268,
        39.319289836649,
        1.2066173678828,
    };
    static double const expected_costheta[] = {
        0.99992128683238,
        0.97331314773255,
        0.9996196536095,
        0.99925389709579,
    };
    EXPECT_VEC_SOFT_EQ(expected_pair_energy, pair_energy);
    EXPECT_VEC_SOFT_EQ(expected_costheta, costheta);

    // Next sample should fail because we're out of secondary buffer space
    {
        Interaction result = interact(rng);
        EXPECT_EQ(0, result.secondaries.size());
        EXPECT_EQ(Action::failed, result.action);
    }
}

TEST_F(MuPairProductionTest, stress_test)
{
    unsigned int const num_samples = 10000;
    std::vector<double> avg_engine_samples;
    std::vector<double> avg_electron_energy;
    std::vector<double> avg_positron_energy;
    std::vector<double> avg_costheta;

    // Get view to the current element
    auto element = this->material_track().material_record().element_record(
        ElementComponentId{0});

    // Get the production cuts
    auto cutoff = this->cutoff_params()->get(MaterialId{0});

    for (real_type inc_e : {1e3, 1e4, 1e5, 1e6, 1e7})
    {
        SCOPED_TRACE("Incident energy: " + std::to_string(inc_e));
        this->set_inc_particle(pdg::mu_minus(), MevEnergy{inc_e});

        RandomEngine& rng = InteractorHostBase::rng();
        RandomEngine::size_type num_particles_sampled = 0;
        double electron_energy = 0;
        double positron_energy = 0;
        double costheta = 0;

        // Loop over several incident directions
        for (Real3 const& inc_dir :
             {Real3{0, 0, 1}, Real3{1, 0, 0}, Real3{1e-9, 0, 1}, Real3{1, 1, 1}})
        {
            SCOPED_TRACE("Incident direction: " + to_string(inc_dir));
            this->set_inc_direction(inc_dir);
            this->resize_secondaries(2 * num_samples);

            // Create the interactor
            MuPairProductionInteractor interact(model_->host_ref(),
                                                this->particle_track(),
                                                cutoff,
                                                element,
                                                this->direction(),
                                                this->secondary_allocator());

            // Loop over many particles
            for (unsigned int i = 0; i < num_samples; ++i)
            {
                Interaction result = interact(rng);
                this->sanity_check(result);

                electron_energy
                    += value_as<MevEnergy>(result.secondaries[0].energy);
                positron_energy
                    += value_as<MevEnergy>(result.secondaries[1].energy);
                costheta += dot_product(result.secondaries[0].direction,
                                        result.secondaries[1].direction);
            }
            EXPECT_EQ(2 * num_samples,
                      this->secondary_allocator().get().size());
            num_particles_sampled += num_samples;
        }
        avg_engine_samples.push_back(real_type(rng.count())
                                     / num_particles_sampled);
        avg_electron_energy.push_back(electron_energy / num_particles_sampled);
        avg_positron_energy.push_back(positron_energy / num_particles_sampled);
        avg_costheta.push_back(costheta / num_particles_sampled);
    }

    // Gold values for average number of calls to RNG
    static double const expected_avg_engine_samples[] = {10, 10, 10, 10, 10};
    static double const expected_avg_electron_energy[] = {
        5.9874014528792,
        20.788005133512,
        98.175982053115,
        555.88642035635,
        2856.0867088461,
    };
    static double const expected_avg_positron_energy[] = {
        5.9071340808566,
        21.495722289587,
        100.9012745799,
        546.91384743321,
        2824.8532482048,
    };
    static double const expected_avg_costheta[] = {
        0.94178280002659,
        0.99880165151033,
        0.99998776687485,
        0.99999983141391,
        0.99999999832285,
    };
    EXPECT_VEC_SOFT_EQ(expected_avg_engine_samples, avg_engine_samples);
    EXPECT_VEC_SOFT_EQ(expected_avg_electron_energy, avg_electron_energy);
    EXPECT_VEC_SOFT_EQ(expected_avg_positron_energy, avg_positron_energy);
    EXPECT_VEC_SOFT_EQ(expected_avg_costheta, avg_costheta);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
