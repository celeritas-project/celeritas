//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file PhotoelectricInteractor.test.cc
//---------------------------------------------------------------------------//
#include "physics/em/PhotoelectricInteractor.hh"

#include "celeritas_test.hh"
#include "base/ArrayUtils.hh"
#include "base/Range.hh"
#include "physics/base/Units.hh"
#include "../InteractorHostTestBase.hh"
#include "../InteractionIO.hh"

using celeritas::ElementDefId;
using celeritas::PhotoelectricInteractor;
namespace pdg = celeritas::pdg;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class PhotoelectricInteractorTest
    : public celeritas_test::InteractorHostTestBase
{
    using Base = celeritas_test::InteractorHostTestBase;

  protected:
    void SetUp() override
    {
        using celeritas::MatterState;
        using celeritas::ParticleDef;
        using namespace celeritas::units;
        using namespace celeritas::constants;
        constexpr auto zero   = celeritas::zero_quantity();
        constexpr auto stable = ParticleDef::stable_decay_constant();

        // Set up shared particle data
        Base::set_particle_params(
            {{"electron",
              pdg::electron(),
              MevMass{0.5109989461},
              ElementaryCharge{-1},
              stable},
             {"gamma", pdg::gamma(), zero, zero, stable}});
        const auto& params    = this->particle_params();
        pointers_.electron_id = params.find(pdg::electron());
        pointers_.gamma_id    = params.find(pdg::gamma());
        pointers_.inv_electron_mass
            = 1 / (params.get(pointers_.electron_id).mass.value());

        // Set default particle to incident 10 MeV photon
        this->set_inc_particle(pdg::gamma(), MevEnergy{10});
        this->set_inc_direction({0, 0, 1});

        // Set up shared material data
        MaterialParams::Input inp;
        inp.elements  = {{19, AmuMass{39.0983}, "K"}};
        inp.materials = {{1e-5 * na_avogadro,
                          293.,
                          MatterState::solid,
                          {{ElementDefId{0}, 1.0}},
                          "K"}};

        // Set default material to potassium
        this->set_material_params(inp);
        this->set_material("K");
    }

    void sanity_check(const Interaction& interaction) const
    {
        ASSERT_TRUE(interaction);

        // Check change to parent track
        EXPECT_GT(this->particle_track().energy().value(),
                  interaction.energy.value());
        EXPECT_EQ(celeritas::Action::absorbed, interaction.action);

        // Check secondaries
        ASSERT_LT(2, interaction.secondaries.size());
        if (interaction.secondaries.size() == 1)
        {
            const auto& electron = interaction.secondaries.front();
            EXPECT_TRUE(electron);
            EXPECT_EQ(pointers_.electron_id, electron.def_id);
            EXPECT_GT(this->particle_track().energy().value(),
                      electron.energy.value());
            EXPECT_LT(0, electron.energy.value());
            EXPECT_SOFT_EQ(1.0, celeritas::norm(electron.direction));
        }

        // Check conservation between primary and secondaries
        this->check_conservation(interaction);
    }

  protected:
    celeritas::PhotoelectricInteractorPointers pointers_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(PhotoelectricInteractorTest, basic)
{
    // Reserve 4 secondaries
    this->resize_secondaries(4);

    // Create the interactor
    PhotoelectricInteractor interact(pointers_,
                                     this->particle_track(),
                                     this->direction(),
                                     this->secondary_allocator());
    RandomEngine&           rng_engine = this->rng();

    // Sampled element
    ElementDefId el_id{0};

    std::vector<double> energy_electron;
    std::vector<double> costheta_electron;

    // Produce four samples from the original incident energy/dir
    for (int i : celeritas::range(4))
    {
        Interaction result = interact(rng_engine, el_id);
        SCOPED_TRACE(result);
        this->sanity_check(result);
        EXPECT_EQ(result.secondaries.data(),
                  this->secondary_allocator().get().data() + i);

        // Add actual results to vector
        energy_electron.push_back(result.secondaries.front().energy.value());
        costheta_electron.push_back(celeritas::dot_product(
            result.secondaries.front().direction, this->direction()));
    }

    EXPECT_EQ(4, this->secondary_allocator().get().size());

    // Note: these are "gold" values based on the host RNG.
    /*
    const double expected_energy[]
        = {0.4581502636229, 1.325852509857, 9.837250571445, 0.5250297816972};
    const double expected_costheta[] = {
        -0.0642523962721, 0.6656882878883, 0.9991545931877, 0.07782377978055};
    const double expected_energy_electron[]
        = {9.541849736377, 8.674147490143, 0.1627494285554, 9.474970218303};
    const double expected_costheta_electron[]
        = {0.998962567429, 0.9941635460938, 0.3895748042313, 0.9986216572142};
    EXPECT_VEC_SOFT_EQ(expected_energy, energy);
    EXPECT_VEC_SOFT_EQ(expected_costheta, costheta);
    EXPECT_VEC_SOFT_EQ(expected_energy_electron, energy_electron);
    EXPECT_VEC_SOFT_EQ(expected_costheta_electron, costheta_electron);
    */
    PRINT_EXPECTED(energy_electron);
    PRINT_EXPECTED(costheta_electron);

    // Next sample should fail because we're out of secondary buffer space
    {
        Interaction result = interact(rng_engine, el_id);
        EXPECT_EQ(0, result.secondaries.size());
        EXPECT_EQ(celeritas::Action::failed, result.action);
    }
}
