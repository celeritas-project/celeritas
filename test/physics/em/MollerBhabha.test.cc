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

TEST_F(MollerBhabhaInteractorTest, moller_scattering)
{
    this->resize_secondaries(4);
    RandomEngine& rng_engine = this->rng();

    std::vector<Real3>  sampled_inc_exit_dir;
    std::vector<double> sampled_inc_exit_energy;
    std::vector<double> sampled_inc_energy_dep;
    std::vector<Real3>  sampled_sec_dir;
    std::vector<double> sampled_sec_energy;

    // Selected energies for the incident particle's interactor test [MeV]
    real_type inc_energies[4] = {1, 10, 1e3, 1e5};

    for (int i : celeritas::range(4))
    {
        // Set electrons for testing Moller scattering
        this->set_inc_particle(pdg::electron(), MevEnergy{inc_energies[i]});

        MollerBhabhaInteractor mb_interactor(pointers_,
                                             this->particle_track(),
                                             this->direction(),
                                             this->secondary_allocator());

        Interaction result = mb_interactor(rng_engine);
        this->sanity_check(result);

        sampled_inc_exit_dir.push_back(result.direction);
        sampled_inc_exit_energy.push_back(result.energy.value());
        sampled_inc_energy_dep.push_back(result.energy_deposition.value());
        auto secondary = result.secondaries[0];
        sampled_sec_dir.push_back(secondary.direction);
        sampled_sec_energy.push_back(secondary.energy.value());
    }

    //// PRIMARY
    // Incident particle's exiting directions after each interactor call
    Real3 expect_inc_exit_dir[4] = {{-5.96444286125940581e-02,
                                     1.18181394144216134e-02,
                                     9.98149725099525487e-01},
                                    {-2.99334686070503678e-03,
                                     6.16069739754576402e-04,
                                     9.99995330155419859e-01},
                                    {-2.89184561174865401e-05,
                                     -2.38811464542290615e-05,
                                     9.99999999296706688e-01},
                                    {3.99574796587882427e-07,
                                     9.19715146589402801e-09,
                                     9.99999999999920175e-01}};

    // Incident particle's final energy after each interactor call [MeV]
    double expect_inc_exit_energy[4] = {9.92711691664473928e-01,
                                        9.99899283170694986e+00,
                                        9.99998622284976022e+02,
                                        9.99999984369220620e+04};

    // Incident particle's deposited energy after each interactor call [MeV]
    double expect_inc_energy_dep[4] = {0, 0, 0, 0};

    //// SECONDARY
    // Secondary directions for each interactor call
    Real3 expect_sec_dir[4] = {{9.73881762790434480e-01,
                                -1.92968072853493350e-01,
                                1.19656320198327393e-01},
                               {9.78938880987589188e-01,
                                -2.01478361750435653e-01,
                                3.29414182622315352e-02},
                               {7.70546919033302635e-01,
                                6.36325250163066403e-01,
                                3.67099655842354142e-02},
                               {-9.98971559808914056e-01,
                                -2.29936743361696394e-02,
                                3.90783013124576667e-02}};

    // Secondary energies for each interactor call [MeV]
    real_type expect_sec_energy[4] = {7.28830833552605601e-03,
                                      1.00716829305078732e-03,
                                      1.37771502392675734e-03,
                                      1.56307793755843471e-03};

    for (int i : celeritas::range(4))
    {
        EXPECT_VEC_SOFT_EQ(expect_inc_exit_dir[i], sampled_inc_exit_dir[i]);
        EXPECT_VEC_SOFT_EQ(expect_sec_dir[i], sampled_sec_dir[i]);
    }
    EXPECT_VEC_SOFT_EQ(expect_inc_exit_energy, sampled_inc_exit_energy);
    EXPECT_VEC_SOFT_EQ(expect_inc_energy_dep, sampled_inc_energy_dep);
    EXPECT_VEC_SOFT_EQ(expect_sec_energy, sampled_sec_energy);
}

//---------------------------------------------------------------------------//
TEST_F(MollerBhabhaInteractorTest, bhabha_scattering)
{
    this->resize_secondaries(4);
    RandomEngine& rng_engine = this->rng();

    std::vector<Real3>  sampled_inc_exit_dir;
    std::vector<double> sampled_inc_exit_energy;
    std::vector<double> sampled_inc_energy_dep;
    std::vector<Real3>  sampled_sec_dir;
    std::vector<double> sampled_sec_energy;

    // Selected energies for the incident particle's interactor test [MeV]
    real_type inc_energies[4] = {1, 10, 1e3, 1e5};

    for (int i : celeritas::range(4))
    {
        // Set 10 MeV electron for testing Bhabha scattering
        this->set_inc_particle(pdg::positron(), MevEnergy{inc_energies[i]});

        // Create interactor
        MollerBhabhaInteractor mb_interactor(pointers_,
                                             this->particle_track(),
                                             this->direction(),
                                             this->secondary_allocator());

        Interaction result = mb_interactor(rng_engine);
        this->sanity_check(result);

        sampled_inc_exit_dir.push_back(result.direction);
        sampled_inc_exit_energy.push_back(result.energy.value());
        sampled_inc_energy_dep.push_back(result.energy_deposition.value());
        auto secondary = result.secondaries[0];
        sampled_sec_dir.push_back(secondary.direction);
        sampled_sec_energy.push_back(secondary.energy.value());
    }

    //// PRIMARY
    // Incident particle's exiting directions after each interactor call
    Real3 expect_inc_exit_dir[4] = {{-5.98339146337106900e-02,
                                     1.18556847856613259e-02,
                                     9.98137939063468704e-01},
                                    {6.19406177347736359e-03,
                                     1.89364553704606886e-03,
                                     9.99979023632659225e-01},
                                    {-7.35997559188817274e-05,
                                     1.63669752943484403e-06,
                                     9.99999997290198617e-01},
                                    {-3.22691294854657801e-07,
                                     3.84815259626114375e-08,
                                     9.99999999999947153e-01}};

    // Incident particle's final energy after each interactor call [MeV]
    double expect_inc_exit_energy[4] = {9.92665477497147508e-01,
                                        9.99547740388010375e+00,
                                        9.99994691659400587e+02,
                                        9.99999989666164765e+04};

    // Incident particle's deposited energy after each interactor call [MeV]
    double expect_inc_energy_dep[4] = {0, 0, 0, 0};

    //// SECONDARY
    // Secondary directions for each interactor call
    Real3 expect_sec_dir[4] = {{9.73837231712127771e-01,
                                -1.92959249322042975e-01,
                                1.20032388263890455e-01},
                               {-9.53982991975042882e-01,
                                -2.91651213894371375e-01,
                                6.96851523373852172e-02},
                               {9.97163856758067957e-01,
                                -2.21747422993688603e-02,
                                7.19202584765359088e-02},
                               {9.92462814088369005e-01,
                                -1.18353002253964334e-01,
                                3.17825346577122123e-02}};

    // Secondary energies for each interactor call [MeV]
    real_type expect_sec_energy[4] = {7.33452250285249879e-03,
                                      4.52259611989608724e-03,
                                      5.30834059944587742e-03,
                                      1.03338351935111108e-03};

    for (int i : celeritas::range(4))
    {
        EXPECT_VEC_SOFT_EQ(expect_inc_exit_dir[i], sampled_inc_exit_dir[i]);
        EXPECT_VEC_SOFT_EQ(expect_sec_dir[i], sampled_sec_dir[i]);
    }
    EXPECT_VEC_SOFT_EQ(expect_inc_exit_energy, sampled_inc_exit_energy);
    EXPECT_VEC_SOFT_EQ(expect_inc_energy_dep, sampled_inc_energy_dep);
    EXPECT_VEC_SOFT_EQ(expect_sec_energy, sampled_sec_energy);
}
