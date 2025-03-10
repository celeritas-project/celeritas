//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/MuBremsstrahlung.test.cc
//---------------------------------------------------------------------------//
#include "corecel/cont/Range.hh"
#include "corecel/math/ArrayUtils.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/em/interactor/MuBremsstrahlungInteractor.hh"
#include "celeritas/mat/MaterialTrackView.hh"
#include "celeritas/mat/MaterialView.hh"
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

class MuBremsstrahlungTest : public InteractorHostTestBase
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

        // Set 1 keV gamma cutoff
        CutoffParams::Input cut_inp;
        cut_inp.materials = this->material_params();
        cut_inp.particles = this->particle_params();
        cut_inp.cutoffs = {{pdg::gamma(), {{MevEnergy{0.001}, 0.1234}}}};
        this->set_cutoff_params(cut_inp);

        // Set model data
        auto const& params = this->particle_params();
        data_.gamma = params->find(pdg::gamma());
        data_.mu_minus = params->find(pdg::mu_minus());
        data_.mu_plus = params->find(pdg::mu_plus());
        data_.electron_mass = params->get(params->find(pdg::electron())).mass();

        // Set default particle to muon with energy of 1100 MeV
        this->set_inc_particle(pdg::mu_minus(), MevEnergy{1100});
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

        auto const& gamma = interaction.secondaries.front();
        EXPECT_TRUE(gamma);
        EXPECT_EQ(data_.gamma, gamma.particle_id);
        EXPECT_GT(this->particle_track().energy().value(),
                  gamma.energy.value());
        EXPECT_LT(0, gamma.energy.value());
        EXPECT_SOFT_EQ(1.0, norm(gamma.direction));

        // Check conservation between primary and secondaries
        // To be determined: Not sure if momentum is conserved.
        // this->check_conservation(interaction);
        this->check_energy_conservation(interaction);
    }

  protected:
    MuBremsstrahlungData data_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(MuBremsstrahlungTest, dcs)
{
    auto particle = this->particle_track();
    auto element = this->material_track().material_record().element_record(
        ElementComponentId{0});

    MuBremsDiffXsCalculator calc_dcs(
        element, particle.energy(), particle.mass(), data_.electron_mass);

    std::vector<real_type> dcs;
    for (real_type gamma_energy :
         {1e-3, 5e-3, 1e-2, 5e-2, 0.1, 1.0, 10.0, 1e2, 1e3, 1e4})
    {
        dcs.push_back(calc_dcs(MevEnergy{gamma_energy})
                      / ipow<2>(units::centimeter));
    }

    // Note: these are "gold" differential cross sections by the photon energy
    static double const expected_dcs[] = {
        0.004767766673037,
        0.00095316629837454,
        0.00047634214548039,
        9.4889763823363e-05,
        4.7216419355064e-05,
        4.4118702152857e-06,
        3.4059618626578e-07,
        1.8880368752007e-08,
        1.2292516852155e-10,
        0,
    };
    EXPECT_VEC_SOFT_EQ(expected_dcs, dcs);
}

TEST_F(MuBremsstrahlungTest, basic)
{
    // Reserve 4 secondaries
    int num_samples = 4;
    this->resize_secondaries(num_samples);

    auto material = this->material_track().material_record();

    // Create the interactor
    MuBremsstrahlungInteractor interact(
        data_,
        this->particle_track(),
        this->direction(),
        this->cutoff_params()->get(MaterialId{0}),
        this->secondary_allocator(),
        material,
        ElementComponentId{0});
    RandomEngine& rng_engine = this->rng();

    std::vector<double> energy;
    std::vector<double> costheta;

    // Produce four samples from the original incident energy
    for (int i : range(num_samples))
    {
        Interaction result = interact(rng_engine);
        SCOPED_TRACE(result);
        this->sanity_check(result);

        EXPECT_EQ(result.secondaries.data(),
                  this->secondary_allocator().get().data() + i);

        energy.push_back(result.secondaries[0].energy.value());
        costheta.push_back(dot_product(result.secondaries.front().direction,
                                       this->direction()));
    }

    EXPECT_EQ(num_samples, this->secondary_allocator().get().size());

    // Note: these are "gold" values based on the host RNG.
    static double const expected_energy[] = {
        0.0065836962047077,
        0.072737465588407,
        0.0046101810148469,
        0.0047801585825437,
    };
    static double const expected_costheta[] = {
        0.89331064584376,
        0.99911225062766,
        0.9983850611905,
        0.97306765528221,
    };

    EXPECT_VEC_SOFT_EQ(expected_energy, energy);
    EXPECT_VEC_SOFT_EQ(expected_costheta, costheta);

    // Next sample should fail because we're out of secondary buffer space
    {
        Interaction result = interact(rng_engine);
        EXPECT_EQ(0, result.secondaries.size());
        EXPECT_EQ(Action::failed, result.action);
    }
}

TEST_F(MuBremsstrahlungTest, stress_test)
{
    unsigned int const num_samples = 10000;
    std::vector<real_type> avg_engine_samples;
    std::vector<real_type> avg_energy;
    std::vector<real_type> avg_costheta;

    for (auto particle : {pdg::mu_minus(), pdg::mu_plus()})
    {
        for (real_type inc_e : {1e-2, 0.1, 1.0, 1e2, 1e4, 1e6, 1e8})
        {
            SCOPED_TRACE("Incident energy: " + std::to_string(inc_e));
            this->set_inc_particle(particle, MevEnergy{inc_e});

            RandomEngine& rng_engine = this->rng();
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

                auto material = this->material_track().material_record();

                // Create interactor
                MuBremsstrahlungInteractor interact(
                    data_,
                    this->particle_track(),
                    this->direction(),
                    this->cutoff_params()->get(MaterialId{0}),
                    this->secondary_allocator(),
                    material,
                    ElementComponentId{0});

                for (unsigned int i = 0; i < num_samples; i++)
                {
                    Interaction result = interact(rng_engine);
                    this->sanity_check(result);

                    energy += result.secondaries.front().energy.value();
                    costheta += dot_product(
                        result.direction, result.secondaries.front().direction);
                }
                EXPECT_EQ(num_samples,
                          this->secondary_allocator().get().size());
                num_particles_sampled += num_samples;
            }
            avg_engine_samples.push_back(real_type(rng_engine.count())
                                         / num_particles_sampled);
            avg_energy.push_back(energy / num_particles_sampled);
            avg_costheta.push_back(costheta / num_particles_sampled);
        }
    }

    // Gold values for average number of calls to RNG
    static double const expected_avg_engine_samples[] = {
        8.1181,
        8.4645,
        9.089,
        10.2864,
        8.6038,
        8.1621,
        8.1053,
        8.1199,
        8.4714,
        9.0945,
        10.259,
        8.5959,
        8.1658,
        8.1053,
    };
    static double const expected_avg_energy[] = {
        0.0038567895491616,
        0.019186468482337,
        0.098553972746449,
        1.6852671846735,
        229.53138001979,
        33432.342774211,
        3131579.3398501,
        0.003867632996639,
        0.018870518265507,
        0.098670522149155,
        1.670211265977,
        229.71614742786,
        33171.81427539,
        2988310.0247441,
    };
    static double const expected_avg_costheta[] = {
        0.66306119119239,
        0.66271159718346,
        0.66305830115835,
        0.80473787144944,
        0.9994439910641,
        0.99999981472127,
        0.99999999993925,
        0.66247296093943,
        0.6640155768717,
        0.66170431997873,
        0.80556733172333,
        0.99942134970672,
        0.99999965236971,
        0.99999999995865,
    };

    EXPECT_VEC_SOFT_EQ(expected_avg_engine_samples, avg_engine_samples);
    EXPECT_VEC_SOFT_EQ(expected_avg_energy, avg_energy);
    EXPECT_VEC_SOFT_EQ(expected_avg_costheta, avg_costheta);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
