//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file PhotoelectricInteractor.test.cc
//---------------------------------------------------------------------------//
#include "physics/em/PhotoelectricInteractor.hh"

#include <fstream>
#include "celeritas_test.hh"
#include "base/ArrayUtils.hh"
#include "base/Range.hh"
#include "io/LivermoreParamsReader.hh"
#include "physics/base/Units.hh"
#include "physics/em/LivermoreParams.hh"
#include "../InteractorHostTestBase.hh"
#include "../InteractionIO.hh"

using celeritas::ElementDefId;
using celeritas::LivermoreParams;
using celeritas::LivermoreParamsReader;
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
    void set_livermore_params(LivermoreParams::Input inp)
    {
        REQUIRE(!inp.elements.empty());

        livermore_params_ = std::make_shared<LivermoreParams>(std::move(inp));
        data_             = livermore_params_->host_pointers();
    }

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

        // Set Livermore photoelectric data
        LivermoreParams::Input li;
        std::string data_path = this->test_data_path("physics/em", "");
        LivermoreParamsReader read_element_data(data_path.c_str());
        li.elements.push_back(read_element_data(19));
        set_livermore_params(li);

        // Set default particle to incident 10 MeV photon
        this->set_inc_particle(pdg::gamma(), MevEnergy{10});
        this->set_inc_direction({0, 0, 1});

        // Set up shared material data
        MaterialParams::Input mi;
        mi.elements  = {{19, AmuMass{39.0983}, "K"}};
        mi.materials = {{1e-5 * na_avogadro,
                         293.,
                         MatterState::solid,
                         {{ElementDefId{0}, 1.0}},
                         "K"}};

        // Set default material to potassium
        this->set_material_params(mi);
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
        ASSERT_GT(2, interaction.secondaries.size());
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
        // TODO: atomic relaxation has not been implemented. If the binding
        // energy of the sampled shell is larger than the incident photon
        // energy, or there is an energy balance following atomic deexcitation,
        // the energy will be deposited locally.
        // this->check_conservation(interaction);
    }

  protected:
    std::shared_ptr<LivermoreParams>           livermore_params_;
    celeritas::PhotoelectricInteractorPointers pointers_;
    celeritas::LivermoreParamsPointers         data_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(PhotoelectricInteractorTest, basic)
{
    // Reserve 4 secondaries
    this->resize_secondaries(4);

    // Sampled element
    ElementDefId el_id{0};

    // Create the interactor
    PhotoelectricInteractor interact(pointers_,
                                     data_,
                                     el_id,
                                     this->particle_track(),
                                     this->direction(),
                                     this->secondary_allocator());
    RandomEngine&           rng_engine = this->rng();

    std::vector<double> energy_electron;
    std::vector<double> costheta_electron;

    // Produce four samples from the original incident energy/dir
    for (int i : celeritas::range(4))
    {
        Interaction result = interact(rng_engine);
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
    const double expected_energy_electron[]
        = {9.9964167, 9.9964167, 9.9964167, 9.9964167};
    const double expected_costheta_electron[]
        = {0.9989507514765, 0.9309974252831, 0.9901365221334, 0.9971118554745};
    EXPECT_VEC_SOFT_EQ(expected_energy_electron, energy_electron);
    EXPECT_VEC_SOFT_EQ(expected_costheta_electron, costheta_electron);

    // Next sample should fail because we're out of secondary buffer space
    {
        Interaction result = interact(rng_engine);
        EXPECT_EQ(0, result.secondaries.size());
        EXPECT_EQ(celeritas::Action::failed, result.action);
    }
}

TEST_F(PhotoelectricInteractorTest, stress_test)
{
    RandomEngine& rng_engine = this->rng();

    const int           num_samples = 8192;
    std::vector<double> avg_engine_samples;

    ElementDefId el_id{0};

    for (double inc_e : {0.0001, 0.01, 1.0, 10.0, 1000.0})
    {
        SCOPED_TRACE("Incident energy: " + std::to_string(inc_e));
        this->set_inc_particle(pdg::gamma(), MevEnergy{inc_e});
        RandomEngine::size_type num_particles_sampled = 0;

        // Loop over several incident directions (shouldn't affect anything
        // substantial, but scattering near Z axis loses precision)
        for (const Real3& inc_dir :
             {Real3{0, 0, 1}, Real3{1, 0, 0}, Real3{1e-9, 0, 1}, Real3{1, 1, 1}})
        {
            SCOPED_TRACE("Incident direction: " + to_string(inc_dir));
            this->set_inc_direction(inc_dir);
            this->resize_secondaries(num_samples);

            // Create interactor
            PhotoelectricInteractor interact(pointers_,
                                             data_,
                                             el_id,
                                             this->particle_track(),
                                             this->direction(),
                                             this->secondary_allocator());

            // Loop over many particles
            for (int i = 0; i < num_samples; ++i)
            {
                Interaction result = interact(rng_engine);
                // SCOPED_TRACE(result);
                this->sanity_check(result);
            }
            EXPECT_EQ(num_samples, this->secondary_allocator().get().size());
            num_particles_sampled += num_samples;
        }
        avg_engine_samples.push_back(double(rng_engine.count())
                                     / double(num_particles_sampled));
        rng_engine.reset_count();
    }
    // PRINT_EXPECTED(avg_engine_samples);
    // Gold values for average number of calls to RNG
    const double expected_avg_engine_samples[]
        = {15.99755859375, 16.09204101562, 13.79919433594, 8.590209960938, 2};
    EXPECT_VEC_SOFT_EQ(expected_avg_engine_samples, avg_engine_samples);
}
