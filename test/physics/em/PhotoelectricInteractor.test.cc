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
#include "physics/base/Units.hh"
#include "physics/em/LivermoreParams.hh"
#include "../InteractorHostTestBase.hh"
#include "../InteractionIO.hh"

using celeritas::ElementDefId;
using celeritas::LivermoreParams;
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
    LivermoreParams::ElementInput
    read_livermore_element(int atomic_number, ElementDefId el_id)
    {
        using celeritas::ImportPhysicsVector;
        using celeritas::ImportPhysicsVectorType;
        REQUIRE(atomic_number > 0 && atomic_number < 101);

        LivermoreParams::ElementInput result;
        result.el_id = el_id;

        // TODO: check units
        // Read photoelectric effect total cross section above K shell energy
        // but below parameterized values
        {
            std::ostringstream os;
            os << "pe-cs-" << atomic_number << ".dat";
            std::string filename
                = this->test_data_path("physics/em", os.str().c_str());
            std::ifstream infile(filename.c_str());
            CHECK(infile.is_open());

            // Set the physics vector type and the data type
            result.xs_high.vector_type
                = ImportPhysicsVectorType::low_energy_free;
            result.xs_high.data_type = ImportPhysicsVector::DataType::xs;

            // Read tabulated energies and cross sections
            double       energy_min, energy_max;
            unsigned int size;
            infile >> energy_min >> energy_max >> size;
            if (!infile.fail())
            {
                infile >> size;
                result.xs_high.energy.resize(size);
                result.xs_high.xs_eloss.resize(size);
                for (auto i : celeritas::range(size))
                {
                    infile >> result.xs_high.energy[i]
                        >> result.xs_high.xs_eloss[i];
                }
            }
            infile.close();
        }

        // Read photoelectric effect total cross section below K shell energy
        {
            std::ostringstream os;
            os << "pe-le-cs-" << atomic_number << ".dat";
            std::string filename
                = this->test_data_path("physics/em", os.str().c_str());
            std::ifstream infile(filename.c_str());
            CHECK(infile.is_open());

            // Set the physics vector type and the data type
            result.xs_low.vector_type
                = ImportPhysicsVectorType::low_energy_free;
            result.xs_low.data_type = ImportPhysicsVector::DataType::xs;

            // Read tabulated energies and cross sections
            double       energy_min, energy_max;
            unsigned int size;
            infile >> energy_min >> energy_max >> size;
            if (!infile.fail())
            {
                infile >> size;
                result.xs_high.energy.resize(size);
                result.xs_high.xs_eloss.resize(size);
                for (auto i : celeritas::range(size))
                {
                    infile >> result.xs_high.energy[i]
                        >> result.xs_high.xs_eloss[i];
                }
            }
            infile.close();
        }

        // Read subshell cross sections fit parameters in low energy interval
        {
            std::ostringstream os;
            os << "pe-low-" << atomic_number << ".dat";
            std::string filename
                = this->test_data_path("physics/em", os.str().c_str());
            std::ifstream infile(filename.c_str());
            CHECK(infile.is_open());

            // Read the number of subshells and energy threshold
            unsigned int num_shells;
            double       threshold;
            infile >> num_shells >> num_shells >> threshold;
            result.thresh_low = MevEnergy{threshold};
            result.shells.resize(num_shells);

            // Read the binding energies and fit parameters
            for (auto& shell : result.shells)
            {
                double binding_energy;
                infile >> binding_energy;
                shell.binding_energy = MevEnergy{binding_energy};
                shell.param_low.resize(6);
                for (auto i : celeritas::range(6))
                {
                    infile >> shell.param_low[i];
                }
            }
            infile.close();
        }

        // Read subshell cross sections fit parameters in high energy interval
        {
            std::ostringstream os;
            os << "pe-high-" << atomic_number << ".dat";
            std::string filename
                = this->test_data_path("physics/em", os.str().c_str());
            std::ifstream infile(filename.c_str());
            CHECK(infile.is_open());

            // Read the number of subshells and energy threshold
            unsigned int num_shells;
            double       threshold;
            infile >> num_shells >> num_shells >> threshold;
            result.thresh_high = MevEnergy{threshold};
            CHECK(num_shells == result.shells.size());

            // Read the binding energies and fit parameters
            for (auto& shell : result.shells)
            {
                double binding_energy;
                infile >> binding_energy;
                CHECK(binding_energy == shell.binding_energy.value());
                shell.param_high.resize(6);
                for (auto i : celeritas::range(6))
                {
                    infile >> shell.param_high[i];
                }
            }
            infile.close();
        }

        // Read tabulated subshell cross sections
        {
            std::ostringstream os;
            os << "pe-ss-cs-" << atomic_number << ".dat";
            std::string filename
                = this->test_data_path("physics/em", os.str().c_str());
            std::ifstream infile(filename.c_str());
            CHECK(infile.is_open());

            // Read tabulated subshell cross sections
            for (auto& shell : result.shells)
            {
                double       min_energy, max_energy;
                unsigned int size, shell_id;
                infile >> min_energy >> max_energy >> size >> shell_id;
                shell.energy.resize(size);
                shell.xs.resize(size);
                for (auto i : celeritas::range(size))
                {
                    infile >> shell.energy[i] >> shell.xs[i];
                }
            }
            infile.close();
        }

        return result;
    }

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
        li.elements.push_back(read_livermore_element(19, ElementDefId{0}));
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

    // Create the interactor
    PhotoelectricInteractor interact(pointers_,
                                     data_,
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
