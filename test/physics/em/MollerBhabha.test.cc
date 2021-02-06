//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file MollerBhabha.test.cc
//---------------------------------------------------------------------------//
#include "physics/em/detail/MollerBhabhaInteractor.hh"

#include "celeritas_test.hh"
#include "base/ArrayUtils.hh"
#include "base/Range.hh"
#include "base/Types.hh"
#include "physics/base/Units.hh"
#include "../InteractorHostTestBase.hh"
#include "../InteractionIO.hh"
#include "physics/material/MaterialTrackView.hh"

using celeritas::Action;
using celeritas::dot_product;
using celeritas::normalize_direction;
using celeritas::detail::MollerBhabhaInteractor;
using celeritas::units::AmuMass;
namespace constants = celeritas::constants;
namespace pdg       = celeritas::pdg;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class MollerBhabhaInteractorTest : public celeritas_test::InteractorHostTestBase
{
    using Base = celeritas_test::InteractorHostTestBase;

  protected:
    void SetUp() override
    {
        using celeritas::ParticleDef;
        using namespace celeritas::units;
        constexpr auto stable = ParticleDef::stable_decay_constant();

        // Particles needed by interactor
        Base::set_particle_params({{"electron",
                                    pdg::electron(),
                                    MevMass{0.5109989461},
                                    ElementaryCharge{-1},
                                    stable},

                                   {"positron",
                                    pdg::positron(),
                                    MevMass{0.5109989461},
                                    ElementaryCharge{1},
                                    stable}});

        const auto& params           = this->particle_params();
        pointers_.electron_id        = params.find(pdg::electron());
        pointers_.positron_id        = params.find(pdg::positron());
        pointers_.electron_mass_c_sq = 0.5109989461;
        pointers_.min_valid_energy   = 1e-3; // [MeV]

        // Set default incident direction. Particle is defined in the tests
        this->set_inc_direction({0, 0, 1});

        // Setup MaterialView
        MaterialParams::Input inp;
        inp.elements  = {{29, AmuMass{63.546}, "Cu"}};
        inp.materials = {
            {1.0 * constants::na_avogadro,
             293.0,
             celeritas::MatterState::solid,
             {{celeritas::ElementId{0}, 1.0}},
             "Cu"},
        };
        this->set_material_params(inp);
        this->set_material("Cu");
    }

    void sanity_check(const Interaction& interaction) const
    {
        ASSERT_TRUE(interaction);

        // Check change to parent track
        EXPECT_GT(this->particle_track().energy().value(),
                  interaction.energy.value());
        EXPECT_LT(0, interaction.energy.value());
        EXPECT_SOFT_EQ(1.0, celeritas::norm(interaction.direction));
        EXPECT_EQ(celeritas::Action::scattered, interaction.action);

        // Check secondaries
        ASSERT_EQ(1, interaction.secondaries.size());
        const auto& electron = interaction.secondaries.front();
        EXPECT_TRUE(electron);
        EXPECT_EQ(pointers_.electron_id, electron.particle_id);
        EXPECT_GT(this->particle_track().energy().value(),
                  electron.energy.value());
        EXPECT_LT(0, electron.energy.value());
        EXPECT_SOFT_EQ(1.0, celeritas::norm(electron.direction));

        // Check conservation between primary and secondaries
        this->check_conservation(interaction);
    }

  protected:
    celeritas::detail::MollerBhabhaPointers pointers_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(MollerBhabhaInteractorTest, basic)
{
    // Sample 4 Moller and 4 Bhabha interactors
    this->resize_secondaries(8);
    RandomEngine& rng_engine = this->rng();

    // Sampled Moller results
    std::vector<double> m_sampled_inc_exit_cost;
    std::vector<double> m_sampled_inc_exit_e;
    std::vector<double> m_sampled_inc_edep;
    std::vector<double> m_sampled_sec_cost;
    std::vector<double> m_sampled_sec_e;

    // Sampled Bhabha results
    std::vector<double> b_sampled_inc_exit_cost;
    std::vector<double> b_sampled_inc_exit_e;
    std::vector<double> b_sampled_inc_edep;
    std::vector<double> b_sampled_sec_cost;
    std::vector<double> b_sampled_sec_e;

    // Selected energies for the incident particle's interactor test [MeV]
    real_type inc_energies[4] = {1, 10, 1e3, 1e5};

    Real3 dir0 = {5, 5, 5};
    Real3 dir1 = {-3, 7, 10};
    Real3 dir2 = {1, -10, 5};
    Real3 dir3 = {3, 7, -6};
    normalize_direction(&dir0);
    normalize_direction(&dir1);
    normalize_direction(&dir2);
    normalize_direction(&dir3);

    // Selected directions for the incident particle's interactor test
    Real3 inc_direction[4] = {dir0, dir1, dir2, dir3};

    for (int i : celeritas::range(4))
    {
        this->set_inc_direction(inc_direction[i]);

        //// Sample Moller
        this->set_inc_particle(pdg::electron(), MevEnergy{inc_energies[i]});

        MollerBhabhaInteractor m_interactor(pointers_,
                                            this->particle_track(),
                                            this->direction(),
                                            this->secondary_allocator());

        Interaction m_result = m_interactor(rng_engine);
        this->sanity_check(m_result);

        m_sampled_inc_exit_cost.push_back(
            dot_product(m_result.direction, this->direction()));
        m_sampled_inc_exit_e.push_back(m_result.energy.value());
        m_sampled_inc_edep.push_back(m_result.energy_deposition.value());
        EXPECT_EQ(1, m_result.secondaries.size());
        m_sampled_sec_cost.push_back(
            dot_product(m_result.secondaries[0].direction, this->direction()));
        m_sampled_sec_e.push_back(m_result.secondaries[0].energy.value());

        //// Sample Bhabha
        this->set_inc_particle(pdg::positron(), MevEnergy{inc_energies[i]});

        MollerBhabhaInteractor b_interactor(pointers_,
                                            this->particle_track(),
                                            this->direction(),
                                            this->secondary_allocator());

        Interaction b_result = b_interactor(rng_engine);
        this->sanity_check(b_result);

        b_sampled_inc_exit_cost.push_back(
            dot_product(b_result.direction, this->direction()));
        b_sampled_inc_exit_e.push_back(b_result.energy.value());
        b_sampled_inc_edep.push_back(b_result.energy_deposition.value());
        EXPECT_EQ(1, b_result.secondaries.size());
        b_sampled_sec_cost.push_back(
            dot_product(b_result.secondaries[0].direction, this->direction()));
        b_sampled_sec_e.push_back(b_result.secondaries[0].energy.value());
    }

    //// Moller test
    // Gold values based on the host rng. Energies are in MeV
    // clang-format off
    const double m_expect_inc_exit_cost[] = {
        9.9814972509953e-01,
        9.9999361233299e-01,
        9.9999999546102e-01,
        9.9999999999976e-01
    };
    const double m_expect_inc_exit_e[] = {
        9.9271169166447e-01,
        9.9986223880045e+00,
        9.9999110844690e+02,
        9.9999995281335e+04
    };
    const double m_expect_inc_edep[] = {0, 0, 0, 0};
    const double m_expect_sec_cost[] = {
        1.1965632019833e-01,
        3.8519098201884e-02,
        9.2919010737665e-02,
        6.7793258423643e-02
    };
    const double m_expect_sec_e[] = {
        7.2883083355261e-03,
        1.3776119954606e-03,
        8.8915531042938e-03,
        4.7186649798110e-03
    };
    // clang-format on

    EXPECT_VEC_SOFT_EQ(m_expect_inc_exit_cost, m_sampled_inc_exit_cost);
    EXPECT_VEC_SOFT_EQ(m_expect_inc_exit_e, m_sampled_inc_exit_e);
    EXPECT_VEC_SOFT_EQ(m_expect_inc_edep, m_sampled_inc_edep);
    EXPECT_VEC_SOFT_EQ(m_expect_sec_cost, m_sampled_sec_cost);
    EXPECT_VEC_SOFT_EQ(m_expect_sec_e, m_sampled_sec_e);

    //// Bhabha test
    // Gold values based on the host rng. Energies are in MeV
    // clang-format off
    const double b_expect_inc_exit_cost[] = {
        9.9974531079032e-01,
        9.9999419049901e-01,
        9.9999999898647e-01,
        9.9999999999993e-01
    };
    const double b_expect_inc_exit_e[] = {
        9.9899283748381e-01,
        9.9987470650722e+00,
        9.9999801454613e+02,
        9.9999998649830e+04
    };
    const double b_expect_inc_edep[] = {0, 0, 0, 0};
    const double b_expect_sec_cost[] = {
        4.4617089494519e-02,
        3.6736972883451e-02,
        4.4056020441594e-02,
        3.6323260625147e-02
    };
    const double b_expect_sec_e[] = {
        1.0071625161866e-03,
        1.2529349277679e-03,
        1.9854538738138e-03,
        1.3501704133594e-03
    };
    // clang-format on

    EXPECT_VEC_SOFT_EQ(b_expect_inc_exit_cost, b_sampled_inc_exit_cost);
    EXPECT_VEC_SOFT_EQ(b_expect_inc_exit_e, b_sampled_inc_exit_e);
    EXPECT_VEC_SOFT_EQ(b_expect_inc_edep, b_sampled_inc_edep);
    EXPECT_VEC_SOFT_EQ(b_expect_sec_cost, b_sampled_sec_cost);
    EXPECT_VEC_SOFT_EQ(b_expect_sec_e, b_sampled_sec_e);
}

TEST_F(MollerBhabhaInteractorTest, stress_test)
{
    RandomEngine& rng = this->rng();

    const int           num_samples = 1e5; // Must be an even number
    std::vector<double> avg_engine_samples;

    // Moller's max energy fraction is 0.5, which leads to E_K > 2e-3
    // Bhabha's max energy fraction is 1.0, which leads to E_K > 1e-3
    // Since this loop encompasses both Moller and Bhabha, the minimum chosen
    // energy is > 2e-3.
    // NOTE: As E_K -> 2e-3, engine_samples -> infinity
    for (double inc_e : {5e-3, 1.0, 10.0, 100.0, 1000.0})
    {
        RandomEngine::size_type num_particles_sampled = 0;

        // Loop over several incident directions (shouldn't affect anything
        // substantial, but scattering near Z axis loses precision)
        for (const Real3& inc_dir :
             {Real3{0, 0, 1}, Real3{1, 0, 0}, Real3{1e-9, 0, 1}, Real3{1, 1, 1}})
        {
            this->set_inc_direction(inc_dir);
            this->resize_secondaries(num_samples);

            // Create Moller interactor
            this->set_inc_particle(pdg::electron(), MevEnergy{inc_e});
            MollerBhabhaInteractor m_interact(pointers_,
                                              this->particle_track(),
                                              this->direction(),
                                              this->secondary_allocator());

            // Loop over half the sample size
            for (int i = 0; i < num_samples / 2; ++i)
            {
                Interaction result = m_interact(rng);
                this->sanity_check(result);
            }

            // Create Bhabha interactor
            this->set_inc_particle(pdg::positron(), MevEnergy{inc_e});
            MollerBhabhaInteractor b_interact(pointers_,
                                              this->particle_track(),
                                              this->direction(),
                                              this->secondary_allocator());

            // Loop over half the sample size
            for (int i = 0; i < num_samples / 2; ++i)
            {
                Interaction result = b_interact(rng);
                this->sanity_check(result);
            }

            EXPECT_EQ(num_samples, this->secondary_allocator().get().size());
            num_particles_sampled += num_samples;
        }
        avg_engine_samples.push_back(double(rng.count())
                                     / double(num_particles_sampled));
        rng.reset_count();
    }
    // PRINT_EXPECTED(avg_engine_samples);
    // Gold values for average number of calls to rng
    const double expected_avg_engine_samples[]
        = {292.29072, 10.9784, 8.35317, 8.11999, 8.1031};
    EXPECT_VEC_SOFT_EQ(expected_avg_engine_samples, avg_engine_samples);
}
